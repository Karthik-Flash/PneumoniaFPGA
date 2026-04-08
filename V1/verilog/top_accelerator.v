`timescale 1ns / 1ps
//==============================================================================
// top_accelerator.v  -- BRAM-Corrected + Timing-Fixed Version
//
// TIMING FIX (resolves WNS=-4.329 ns, TNS=-160 ns):
//   Root cause: the MAC array computed 9 × (8×8 multiply) AND the full 9-input
//   adder tree in a single registered clock stage.
//   Vivado reported 23 logic levels, 9.85 ns logic delay -> 16.78 ns total path,
//   violating the 12.5 ns (80 MHz) budget by 4.33 ns.
//
//   Fix: split into a 2-stage pipeline inside the generate block:
//     Stage 1 (mac_prod_reg): register all 9 independent products (8x8->16 bit).
//       Critical path now: one 8x8 multiply ~5-6 levels, ~2.5 ns logic.
//     Stage 2 (mac_out):     sum 9 products + sign-extended bias -> 20-bit result.
//       Critical path now: balanced 9-input adder tree ~10 levels, ~4.5 ns logic.
//   Both stages comfortably fit in 12.5 ns.
//
//   Side effect: mac_out is now valid 1 extra clock after pixel streaming.
//   Handled by a 2-stage valid pipeline (window_valid_reg -> mac_prod_valid ->
//   mac_valid). Transparent to all downstream logic because they simply wait
//   for the mac_valid strobe before writing to BRAM.
//
// Previous BRAM fix retained:
//   4 separate channel BRAMs (fm_ch0..3) instead of one monolithic array,
//   so Vivado can legally infer each as a distinct 2-port RAMB18E1.
//==============================================================================

module top_accelerator(
    input  wire       clk,
    input  wire       rst,
    input  wire       start,
    input  wire [7:0] pixel_in,
    input  wire       pixel_valid,
    output wire       cancer_detected,
    output wire       result_valid
);

    //--------------------------------------------------------------------------
    // Conv weight ROM  (distributed RAM: 36 + 4 entries, tiny)
    //--------------------------------------------------------------------------
    reg signed [7:0] conv_weights [0:35];
    reg signed [7:0] conv_bias_rom [0:3];

    initial begin
        $readmemh("conv1_weights.mem", conv_weights);
        $readmemh("conv1_bias.mem",    conv_bias_rom);
    end

    //--------------------------------------------------------------------------
    // Feature-map storage: 4 SEPARATE BRAMs, one per conv channel
    // Each holds 784 x 20-bit values (~4 RAMB18E1 total)
    //--------------------------------------------------------------------------
    (* ram_style = "block" *) reg [19:0] fm_ch0 [0:783];
    (* ram_style = "block" *) reg [19:0] fm_ch1 [0:783];
    (* ram_style = "block" *) reg [19:0] fm_ch2 [0:783];
    (* ram_style = "block" *) reg [19:0] fm_ch3 [0:783];

    reg  [9:0]  fm_rd_addr;
    reg  [19:0] fm_rd_data_ch0, fm_rd_data_ch1, fm_rd_data_ch2, fm_rd_data_ch3;

    always @(posedge clk) begin
        fm_rd_data_ch0 <= fm_ch0[fm_rd_addr];
        fm_rd_data_ch1 <= fm_ch1[fm_rd_addr];
        fm_rd_data_ch2 <= fm_ch2[fm_rd_addr];
        fm_rd_data_ch3 <= fm_ch3[fm_rd_addr];
    end

    // pool_ch declared early because it drives the mux below
    reg [1:0]  pool_ch;
    reg [19:0] fm_rd_data;
    always @(*) begin
        case (pool_ch)
            2'd0:    fm_rd_data = fm_rd_data_ch0;
            2'd1:    fm_rd_data = fm_rd_data_ch1;
            2'd2:    fm_rd_data = fm_rd_data_ch2;
            default: fm_rd_data = fm_rd_data_ch3;
        endcase
    end

    //--------------------------------------------------------------------------
    // Pooled-map storage (single BRAM: one write port, one read port)
    //--------------------------------------------------------------------------
    (* ram_style = "block" *) reg [19:0] pooled_maps_bram [0:783];

    reg  [9:0]  pm_wr_addr;
    reg  [19:0] pm_wr_data;
    reg         pm_wr_en;
    wire [9:0]  pm_rd_addr;     // driven by FC1's bram_rd_addr output
    reg  [19:0] pm_rd_data;

    always @(posedge clk) begin
        if (pm_wr_en)
            pooled_maps_bram[pm_wr_addr] <= pm_wr_data;
    end
    always @(posedge clk) begin
        pm_rd_data <= pooled_maps_bram[pm_rd_addr];
    end

    //--------------------------------------------------------------------------
    // Line buffer -> 3x3 sliding window
    //--------------------------------------------------------------------------
    wire [71:0] window_out;
    wire        window_valid;

    reg [71:0]  window_out_reg;
    reg         window_valid_reg;

    //--------------------------------------------------------------------------
    // MAC pipeline Stage 1: register 9 independent 8x8 products per filter
    //   One 8x8 signed multiply -> ~5-6 LUT levels -> ~2.5 ns logic delay.
    //   No summation in this stage, so it easily meets 12.5 ns.
    //--------------------------------------------------------------------------
    reg signed [15:0] mac_prod_reg [0:3][0:8];
    reg               mac_prod_valid;   // valid 1 cycle after window_valid_reg

    //--------------------------------------------------------------------------
    // MAC pipeline Stage 2: sum 9 registered products + bias -> mac_out
    //   Balanced 9-input 20-bit adder tree -> ~10 LUT levels -> ~4.5 ns logic.
    //   Comfortably within 12.5 ns budget.
    //--------------------------------------------------------------------------
    (* keep = "true" *) reg signed [19:0] mac_out [0:3];
    reg                                   mac_valid [0:3]; // valid 1 cycle after mac_prod_valid

    wire [19:0] relu_out [0:3];

    reg  [9:0]  fm_write_count;
    reg         conv_complete;
    reg  [1:0]  conv_drain_counter;

    //--------------------------------------------------------------------------
    // Pool bookkeeping
    //--------------------------------------------------------------------------
    reg [19:0] pool_buffer [0:3];
    wire signed [19:0] pool_result;

    reg [7:0]  pool_count;
    reg        pool_complete;
    reg [3:0]  pool_row, pool_col;
    reg [3:0]  pool_read_state;
    reg [1:0]  pool_within_ch_state;
    reg        pool_done_flag;

    //--------------------------------------------------------------------------
    // FC / FSM wires
    //--------------------------------------------------------------------------
    wire [319:0] fc1_output;
    wire         fc1_done;
    wire [39:0]  fc2_output;
    wire         fc2_done;

    wire [2:0] fsm_state;
    wire       conv_en, pool_en, fc1_en, fc2_en;
    wire       output_valid;

    wire signed [19:0] logit0, logit1;
    reg                cancer_flag;

    integer i;

    //==========================================================================
    // Sub-module instantiations
    //==========================================================================

    line_buffer #(.IMG_WIDTH(28)) u_line_buffer (
        .clk        (clk),
        .rst        (rst),
        .start      (start),
        .pixel_in   (pixel_in),
        .pixel_valid(pixel_valid && conv_en),
        .window_out (window_out),
        .window_valid(window_valid)
    );

    // Pipeline register 0->1: raw window
    always @(posedge clk) begin
        if (rst) begin
            window_out_reg   <= 72'd0;
            window_valid_reg <= 1'b0;
        end else begin
            window_out_reg   <= window_out;
            window_valid_reg <= window_valid;
        end
    end

    //--------------------------------------------------------------------------
    // MAC Stage 1 generate: 9 independent products per filter, registered
    //--------------------------------------------------------------------------
    generate
        genvar fi;
        for (fi = 0; fi < 4; fi = fi + 1) begin : mac_stage1
            always @(posedge clk) begin
                if (rst) begin
                    mac_prod_reg[fi][0] <= 16'sd0;
                    mac_prod_reg[fi][1] <= 16'sd0;
                    mac_prod_reg[fi][2] <= 16'sd0;
                    mac_prod_reg[fi][3] <= 16'sd0;
                    mac_prod_reg[fi][4] <= 16'sd0;
                    mac_prod_reg[fi][5] <= 16'sd0;
                    mac_prod_reg[fi][6] <= 16'sd0;
                    mac_prod_reg[fi][7] <= 16'sd0;
                    mac_prod_reg[fi][8] <= 16'sd0;
                end else begin
                    mac_prod_reg[fi][0] <= $signed({1'b0,window_out_reg[71:64]}) * conv_weights[fi*9+0];
                    mac_prod_reg[fi][1] <= $signed({1'b0,window_out_reg[63:56]}) * conv_weights[fi*9+1];
                    mac_prod_reg[fi][2] <= $signed({1'b0,window_out_reg[55:48]}) * conv_weights[fi*9+2];
                    mac_prod_reg[fi][3] <= $signed({1'b0,window_out_reg[47:40]}) * conv_weights[fi*9+3];
                    mac_prod_reg[fi][4] <= $signed({1'b0,window_out_reg[39:32]}) * conv_weights[fi*9+4];
                    mac_prod_reg[fi][5] <= $signed({1'b0,window_out_reg[31:24]}) * conv_weights[fi*9+5];
                    mac_prod_reg[fi][6] <= $signed({1'b0,window_out_reg[23:16]}) * conv_weights[fi*9+6];
                    mac_prod_reg[fi][7] <= $signed({1'b0,window_out_reg[15:8]})  * conv_weights[fi*9+7];
                    mac_prod_reg[fi][8] <= $signed({1'b0,window_out_reg[7:0]})   * conv_weights[fi*9+8];
                end
            end
        end
    endgenerate

    // Stage-1 valid: 1 clock behind window_valid_reg
    always @(posedge clk) begin
        if (rst) mac_prod_valid <= 1'b0;
        else     mac_prod_valid <= window_valid_reg;
    end

    //--------------------------------------------------------------------------
    // MAC Stage 2 generate: sum 9 sign-extended products + bias -> mac_out
    //--------------------------------------------------------------------------
    generate
        genvar fi2;
        for (fi2 = 0; fi2 < 4; fi2 = fi2 + 1) begin : mac_stage2
            always @(posedge clk) begin
                if (rst) begin
                    mac_out[fi2]   <= 20'sd0;
                    mac_valid[fi2] <= 1'b0;
                end else begin
                    mac_out[fi2] <=
                        $signed({{4{mac_prod_reg[fi2][0][15]}}, mac_prod_reg[fi2][0]}) +
                        $signed({{4{mac_prod_reg[fi2][1][15]}}, mac_prod_reg[fi2][1]}) +
                        $signed({{4{mac_prod_reg[fi2][2][15]}}, mac_prod_reg[fi2][2]}) +
                        $signed({{4{mac_prod_reg[fi2][3][15]}}, mac_prod_reg[fi2][3]}) +
                        $signed({{4{mac_prod_reg[fi2][4][15]}}, mac_prod_reg[fi2][4]}) +
                        $signed({{4{mac_prod_reg[fi2][5][15]}}, mac_prod_reg[fi2][5]}) +
                        $signed({{4{mac_prod_reg[fi2][6][15]}}, mac_prod_reg[fi2][6]}) +
                        $signed({{4{mac_prod_reg[fi2][7][15]}}, mac_prod_reg[fi2][7]}) +
                        $signed({{4{mac_prod_reg[fi2][8][15]}}, mac_prod_reg[fi2][8]}) +
                        $signed({{12{conv_bias_rom[fi2][7]}}, conv_bias_rom[fi2]});
                    mac_valid[fi2] <= mac_prod_valid;
                end
            end
        end
    endgenerate

    //--------------------------------------------------------------------------
    // ReLU (purely combinational, no timing impact)
    //--------------------------------------------------------------------------
    generate
        genvar ri;
        for (ri = 0; ri < 4; ri = ri + 1) begin : relu_array
            relu u_relu (
                .data_in (mac_out[ri]),
                .data_out(relu_out[ri])
            );
        end
    endgenerate

    //--------------------------------------------------------------------------
    // Max-pool (single instance, combinational)
    //--------------------------------------------------------------------------
    max_pool u_max_pool (
        .in0    (pool_buffer[0]),
        .in1    (pool_buffer[1]),
        .in2    (pool_buffer[2]),
        .in3    (pool_buffer[3]),
        .max_out(pool_result)
    );

    //--------------------------------------------------------------------------
    // FC1 (BRAM interface to pooled_maps_bram)
    //--------------------------------------------------------------------------
    fc_layer #(
        .INPUT_SIZE   (784),
        .OUTPUT_SIZE  (16),
        .DATA_IN_WIDTH(20),
        .WEIGHT_MEM   ("fc1_weights.mem"),
        .BIAS_MEM     ("fc1_bias.mem"),
        .APPLY_RELU   (1)
    ) u_fc1 (
        .clk         (clk),
        .rst         (rst),
        .bram_rd_addr(pm_rd_addr),
        .bram_rd_data(pm_rd_data),
        .data_in     (20'b0),
        .start       (fc1_en),
        .data_out    (fc1_output),
        .done        (fc1_done)
    );

    //--------------------------------------------------------------------------
    // FC2 (packed bus from FC1)
    //--------------------------------------------------------------------------
    wire [9:0] fc2_bram_rd_addr_unused;

    fc_layer #(
        .INPUT_SIZE   (16),
        .OUTPUT_SIZE  (2),
        .DATA_IN_WIDTH(320),
        .WEIGHT_MEM   ("fc2_weights.mem"),
        .BIAS_MEM     ("fc2_bias.mem"),
        .APPLY_RELU   (0)
    ) u_fc2 (
        .clk         (clk),
        .rst         (rst),
        .bram_rd_addr(fc2_bram_rd_addr_unused),
        .bram_rd_data(20'h00000),
        .data_in     (fc1_output),
        .start       (fc2_en),
        .data_out    (fc2_output),
        .done        (fc2_done)
    );

    //--------------------------------------------------------------------------
    // FSM controller
    //--------------------------------------------------------------------------
    fsm_control u_fsm (
        .clk        (clk),
        .rst        (rst),
        .start      (start),
        .conv_done  (conv_complete),
        .pool_done  (pool_complete),
        .fc1_done   (fc1_done),
        .fc2_done   (fc2_done),
        .state      (fsm_state),
        .conv_en    (conv_en),
        .pool_en    (pool_en),
        .fc1_en     (fc1_en),
        .fc2_en     (fc2_en),
        .output_valid(output_valid)
    );

    //--------------------------------------------------------------------------
    // Completion edge detection + logging
    //--------------------------------------------------------------------------
    reg conv_complete_prev, pool_complete_prev;

    always @(posedge clk) begin
        if (rst) begin
            conv_complete_prev <= 1'b0;
            pool_complete_prev <= 1'b0;
        end else begin
            conv_complete_prev <= conv_complete;
            pool_complete_prev <= pool_complete;
            if (conv_complete && !conv_complete_prev)
                $display("[%0t] CONV COMPLETE - 784 feature maps written", $time);
            if (pool_complete && !pool_complete_prev)
                $display("[%0t] POOL COMPLETE - 784 pooled values written", $time);
        end
    end

    //--------------------------------------------------------------------------
    // Conv BRAM write logic
    // mac_valid[0] now arrives 2 cycles after window_valid_reg (Stage1+Stage2).
    // fm_write_count simply waits for the strobe -- no other change needed.
    //--------------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst || start) begin
            fm_write_count     <= 10'd0;
            conv_drain_counter <= 2'd0;
        end else if (conv_en) begin
            if (mac_valid[0]) begin
                fm_ch0[fm_write_count] <= relu_out[0];
                fm_ch1[fm_write_count] <= relu_out[1];
                fm_ch2[fm_write_count] <= relu_out[2];
                fm_ch3[fm_write_count] <= relu_out[3];

                if (fm_write_count == 10'd0)
                    $display("[%0t] First parallel write: count=%0d", $time, fm_write_count);
                if (fm_write_count == 10'd100)
                    $display("[%0t] Progress: count=%0d", $time, fm_write_count);

                if (fm_write_count == 10'd783) begin
                    fm_write_count     <= 10'd0;
                    conv_drain_counter <= 2'd1;
                    $display("[%0t] Convolution 784 windows written, starting drain", $time);
                end else begin
                    fm_write_count <= fm_write_count + 10'd1;
                end
            end

            if (conv_drain_counter == 2'd1)
                conv_drain_counter <= 2'd2;
            else if (conv_drain_counter == 2'd2) begin
                conv_drain_counter <= 2'd0;
                $display("[%0t] Convolution drain complete, asserting conv_complete", $time);
            end
        end
    end

    always @(posedge clk) begin
        if (rst || start) begin
            conv_complete <= 1'b0;
        end else if (conv_en && conv_drain_counter == 2'd2) begin
            conv_complete <= 1'b1;
        end else if (pool_en && conv_complete) begin
            conv_complete <= 1'b0;
        end
    end

    //--------------------------------------------------------------------------
    // Pool: read 2x2 blocks from fm_ch0..3 BRAMs, write max to pooled_maps_bram
    //--------------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst || start) begin
            pool_count           <= 8'd0;
            pool_done_flag       <= 1'b0;
            pool_row             <= 4'd0;
            pool_col             <= 4'd0;
            pool_read_state      <= 4'd0;
            pool_ch              <= 2'd0;
            pool_within_ch_state <= 2'd0;
            pm_wr_en             <= 1'b0;
            fm_rd_addr           <= 10'd0;
            for (i = 0; i < 4; i = i + 1)
                pool_buffer[i] <= 20'd0;
        end else if (pool_en) begin
            case (pool_read_state)

                4'd0: begin
                    pool_ch              <= 2'd0;
                    pool_within_ch_state <= 2'd0;
                    pool_read_state      <= 4'd1;
                end

                4'd1, 4'd2, 4'd3, 4'd4: begin
                    case (pool_within_ch_state)
                        2'd0: fm_rd_addr <= ((pool_row*2)   * 28) + (pool_col*2);
                        2'd1: fm_rd_addr <= ((pool_row*2)   * 28) + (pool_col*2 + 1);
                        2'd2: fm_rd_addr <= ((pool_row*2+1) * 28) + (pool_col*2);
                        2'd3: fm_rd_addr <= ((pool_row*2+1) * 28) + (pool_col*2 + 1);
                    endcase

                    if (pool_read_state > 4'd1)
                        pool_buffer[pool_within_ch_state - 1] <= fm_rd_data;

                    pool_within_ch_state <= pool_within_ch_state + 2'd1;
                    pool_read_state      <= pool_read_state + 4'd1;
                end

                4'd5: begin
                    pool_buffer[3] <= fm_rd_data;

                    pm_wr_addr <= (pool_ch * 196) + (pool_row * 14) + pool_col;
                    pm_wr_data <= pool_result;
                    pm_wr_en   <= 1'b1;

                    if (pool_ch == 2'd3)
                        pool_read_state <= 4'd6;
                    else begin
                        pool_ch              <= pool_ch + 2'd1;
                        pool_within_ch_state <= 2'd0;
                        pool_read_state      <= 4'd1;
                    end
                end

                4'd6: begin
                    pm_wr_en <= 1'b0;

                    if (pool_col == 4'd13) begin
                        pool_col <= 4'd0;
                        if (pool_row == 4'd13) begin
                            pool_done_flag  <= 1'b1;
                            pool_row        <= 4'd0;
                            pool_count      <= 8'd0;
                            $display("[%0t] Pooling done: 196 positions x 4 channels = 784 pooled values", $time);
                            pool_read_state <= 4'd0;
                        end else begin
                            pool_row        <= pool_row + 4'd1;
                            pool_count      <= pool_count + 8'd1;
                            pool_read_state <= 4'd0;
                        end
                    end else begin
                        pool_col        <= pool_col + 4'd1;
                        pool_count      <= pool_count + 8'd1;
                        pool_read_state <= 4'd0;
                    end
                end

                default: begin
                    pool_read_state <= 4'd0;
                    pm_wr_en        <= 1'b0;
                end
            endcase
        end else begin
            pm_wr_en        <= 1'b0;
            pool_read_state <= 4'd0;
        end
    end

    always @(posedge clk) begin
        if (rst || start) begin
            pool_complete <= 1'b0;
        end else if (pool_en && pool_done_flag) begin
            pool_complete <= 1'b1;
        end else if (fc1_en && pool_complete) begin
            pool_complete <= 1'b0;
        end
    end

    //--------------------------------------------------------------------------
    // Argmax + output
    //--------------------------------------------------------------------------
    assign logit0 = fc2_output[19:0];
    assign logit1 = fc2_output[39:20];

    always @(posedge clk) begin
        if (rst)
            cancer_flag <= 1'b0;
        else if (output_valid)
            cancer_flag <= (logit1 > logit0) ? 1'b1 : 1'b0;
    end

    assign cancer_detected = cancer_flag;
    assign result_valid    = output_valid;

    always @(posedge clk) begin
        if (output_valid)
            $display("LOGITS: logit0(normal)=%0d logit1(pneumonia)=%0d -> decision=%b",
                     $signed(logit0), $signed(logit1), (logit1 > logit0));
    end

endmodule
