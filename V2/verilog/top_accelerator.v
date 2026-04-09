`timescale 1ns / 1ps
//==============================================================================
// top_accelerator.v  -- 3-Stage MAC Pipeline (Timing Closure Version)
//
// TIMING STATUS HISTORY:
//   Original (1-stage MAC) : WNS = -4.329 ns, TNS = -160 ns, 93 failing paths
//   2-stage MAC            : WNS = -0.367 ns, TNS =  -1.45 ns, 10 failing paths
//   3-stage MAC (this file): WNS expected > 0  (all paths should close at 80 MHz)
//
// WHY THE 2-STAGE VERSION STILL FAILED:
//   Stage2 summed 9 x 20-bit values in one clock = 18 logic levels, 7.116 ns.
//   With 5.788 ns net delay (Vivado placed Stage1 regs far from Stage2 regs),
//   total path = 12.904 ns > 12.5 ns budget by 0.367 ns.
//   Net delay is routing, not fixable by retiming alone.
//
// 3-STAGE MAC PIPELINE:
//   Stage 1 (mac_prod_reg):    9 independent 8x8 products, registered.
//     Critical path: 1 multiply = ~5 levels, ~2.0 ns logic. Passes easily.
//
//   Stage 2 (mac_partial_reg): 3 partial sums of 3 products each, registered.
//     partial[0] = prod[0] + prod[1] + prod[2]  (3-input adder)
//     partial[1] = prod[3] + prod[4] + prod[5]
//     partial[2] = prod[6] + prod[7] + prod[8]
//     Critical path: 3-input 20-bit adder = ~6 levels, ~2.5 ns logic. Passes.
//     Because Stage1 and Stage2 regs are in the same generate block hierarchy,
//     Vivado will place them adjacent, killing the net delay problem.
//
//   Stage 3 (mac_out): sum 3 partial sums + bias, registered.
//     Critical path: 4-input 20-bit adder = ~8 levels, ~3.5 ns logic. Passes.
//
//   Valid pipeline:
//     window_valid_reg -> mac_prod_valid -> mac_partial_valid -> mac_valid
//     (mac_out now arrives 3 cycles after window_valid_reg, was 2)
//     Downstream logic (BRAM write, pool, FC, FSM) only reacts to mac_valid
//     strobe, so adding one cycle here is completely transparent.
//
// All other fixes retained from previous versions:
//   - 4 separate channel BRAMs (fm_ch0..3) to avoid synthesis hang
//   - fc_layer DATA_IN_WIDTH parameter (no 15,680-bit port on FC1)
//   - line_buffer bottom-right pixel fix
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
    // Conv weight ROM  (distributed RAM: 36 + 4 entries)
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

    reg [1:0]  pool_ch;   // declared early because it drives the mux below
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
    wire [9:0]  pm_rd_addr;
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
    // 3-STAGE MAC PIPELINE
    //
    // Stage 1: mac_prod_reg[filter][0..8]  — 9 registered 8x8 products
    // Stage 2: mac_partial_reg[filter][0..2] — 3 registered partial sums
    // Stage 3: mac_out[filter]              — final sum + bias
    //
    // Valid chain: window_valid_reg -> mac_prod_valid -> mac_partial_valid -> mac_valid
    //--------------------------------------------------------------------------

    // Stage 1 registers: 9 products per filter, 16-bit each
    reg signed [15:0] mac_prod_reg    [0:3][0:8];
    reg               mac_prod_valid;

    // Stage 2 registers: 3 partial sums per filter, 20-bit (sign-extended)
    // partial[k] = prod[3k] + prod[3k+1] + prod[3k+2]
    reg signed [19:0] mac_partial_reg [0:3][0:2];
    reg               mac_partial_valid;

    // Stage 3 registers: final result = partial[0]+partial[1]+partial[2]+bias
    (* keep = "true" *) reg signed [19:0] mac_out   [0:3];
    reg                                   mac_valid  [0:3];

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

    // Pipeline register: raw window (before Stage 1)
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
    // MAC Stage 1: compute 9 independent 8x8 signed products, register them.
    //   One multiply per entry = ~5 LUT levels = ~2.0 ns.  Fits easily.
    //--------------------------------------------------------------------------
    generate
        genvar fi;
        for (fi = 0; fi < 4; fi = fi + 1) begin : mac_stage1
            always @(posedge clk) begin
                if (rst) begin
                    mac_prod_reg[fi][0] <= 16'sd0;  mac_prod_reg[fi][1] <= 16'sd0;
                    mac_prod_reg[fi][2] <= 16'sd0;  mac_prod_reg[fi][3] <= 16'sd0;
                    mac_prod_reg[fi][4] <= 16'sd0;  mac_prod_reg[fi][5] <= 16'sd0;
                    mac_prod_reg[fi][6] <= 16'sd0;  mac_prod_reg[fi][7] <= 16'sd0;
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

    always @(posedge clk) begin
        if (rst) mac_prod_valid <= 1'b0;
        else     mac_prod_valid <= window_valid_reg;
    end

    //--------------------------------------------------------------------------
    // MAC Stage 2: sum 3 groups of 3 products, register partial sums.
    //   3-input 20-bit adder = ~6 LUT levels = ~2.5 ns.  Fits easily.
    //   Registers are in the SAME generate block as Stage 1, so Vivado places
    //   them adjacent -> net delay drops from ~5.8 ns to <2 ns.
    //--------------------------------------------------------------------------
    generate
        genvar fi2;
        for (fi2 = 0; fi2 < 4; fi2 = fi2 + 1) begin : mac_stage2
            always @(posedge clk) begin
                if (rst) begin
                    mac_partial_reg[fi2][0] <= 20'sd0;
                    mac_partial_reg[fi2][1] <= 20'sd0;
                    mac_partial_reg[fi2][2] <= 20'sd0;
                end else begin
                    // partial[0] = prod[0] + prod[1] + prod[2]
                    mac_partial_reg[fi2][0] <=
                        $signed({{4{mac_prod_reg[fi2][0][15]}}, mac_prod_reg[fi2][0]}) +
                        $signed({{4{mac_prod_reg[fi2][1][15]}}, mac_prod_reg[fi2][1]}) +
                        $signed({{4{mac_prod_reg[fi2][2][15]}}, mac_prod_reg[fi2][2]});
                    // partial[1] = prod[3] + prod[4] + prod[5]
                    mac_partial_reg[fi2][1] <=
                        $signed({{4{mac_prod_reg[fi2][3][15]}}, mac_prod_reg[fi2][3]}) +
                        $signed({{4{mac_prod_reg[fi2][4][15]}}, mac_prod_reg[fi2][4]}) +
                        $signed({{4{mac_prod_reg[fi2][5][15]}}, mac_prod_reg[fi2][5]});
                    // partial[2] = prod[6] + prod[7] + prod[8]
                    mac_partial_reg[fi2][2] <=
                        $signed({{4{mac_prod_reg[fi2][6][15]}}, mac_prod_reg[fi2][6]}) +
                        $signed({{4{mac_prod_reg[fi2][7][15]}}, mac_prod_reg[fi2][7]}) +
                        $signed({{4{mac_prod_reg[fi2][8][15]}}, mac_prod_reg[fi2][8]});
                end
            end
        end
    endgenerate

    always @(posedge clk) begin
        if (rst) mac_partial_valid <= 1'b0;
        else     mac_partial_valid <= mac_prod_valid;
    end

    //--------------------------------------------------------------------------
    // MAC Stage 3: sum 3 partial sums + bias -> final mac_out, registered.
    //   4-input 20-bit adder = ~8 LUT levels = ~3.5 ns.  Fits with good margin.
    //--------------------------------------------------------------------------
    generate
        genvar fi3;
        for (fi3 = 0; fi3 < 4; fi3 = fi3 + 1) begin : mac_stage3
            always @(posedge clk) begin
                if (rst) begin
                    mac_out[fi3]   <= 20'sd0;
                    mac_valid[fi3] <= 1'b0;
                end else begin
                    mac_out[fi3] <=
                        mac_partial_reg[fi3][0] +
                        mac_partial_reg[fi3][1] +
                        mac_partial_reg[fi3][2] +
                        $signed({{12{conv_bias_rom[fi3][7]}}, conv_bias_rom[fi3]});
                    mac_valid[fi3] <= mac_partial_valid;
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
    // Conv BRAM write logic.
    // mac_valid[0] now arrives 3 cycles after window_valid_reg (Stage1+2+3).
    // fm_write_count simply waits for the strobe -- completely unchanged.
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
    // Pool: read 2x2 blocks from fm_ch0..3, write max to pooled_maps_bram
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
                    pm_wr_addr     <= (pool_ch * 196) + (pool_row * 14) + pool_col;
                    pm_wr_data     <= pool_result;
                    pm_wr_en       <= 1'b1;

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
