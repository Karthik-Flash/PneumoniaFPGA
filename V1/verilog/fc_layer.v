`timescale 1ns / 1ps
//==============================================================================
// fc_layer.v  -- Corrected Version
//
// KEY FIXES vs previous version:
//   1. DATA_IN_WIDTH parameter added.
//      FC1 (INPUT_SIZE=784) was declared with a 784*20=15,680-bit data_in port.
//      Even though FC1 uses BRAM mode and never touches data_in, the
//      always @(*) unpack block synthesized 15,680 bits of fanout logic.
//      Now FC1 is instantiated with DATA_IN_WIDTH=20 (port is trivially small).
//
//   2. FC2 (packed mode) weight-index off-by-one BUG FIXED.
//      Old code:
//        weight_idx <= weight_base + input_idx;   // non-blocking: lag of 1 cycle
//        acc += input[input_idx] * weights[weight_idx];  // reads STALE weight_idx
//      Result: weights[0] used twice per neuron, weights[last] never used,
//              and neuron N's first weight was actually neuron N-1's last weight.
//      Fix: compute the ROM address inline as a combinational expression:
//        acc += input[input_idx] * weights[weight_base + input_idx];
//      weight_base is registered and stable for the whole neuron computation.
//
//   3. input_unpacked always block guarded by generate (USE_BRAM==0) so the
//      large loop is not elaborated at all for FC1.
//
// Timing (unchanged):
//   FC1 (BRAM mode,  784 inputs, 16 neurons):  (784+2)*16  = 12,576 cycles
//   FC2 (packed mode, 16 inputs,  2 neurons):  (16+1)*2+extra ≈  36 cycles
//==============================================================================

module fc_layer #(
    parameter INPUT_SIZE    = 784,
    parameter OUTPUT_SIZE   = 16,
    parameter DATA_IN_WIDTH = 20,       // FC1: 20 (BRAM mode, data_in unused)
                                        // FC2: 320 (16*20, packed bus from FC1)
    parameter WEIGHT_MEM    = "fc1_weights.mem",
    parameter BIAS_MEM      = "fc1_bias.mem",
    parameter APPLY_RELU    = 1
)(
    input  wire                       clk,
    input  wire                       rst,
    output reg  [9:0]                 bram_rd_addr,   // → pooled_maps_bram (FC1)
    input  wire [19:0]                bram_rd_data,   // ← pooled_maps_bram (FC1)
    input  wire [DATA_IN_WIDTH-1:0]   data_in,        // packed inputs (FC2 only)
    input  wire                       start,
    output reg  [OUTPUT_SIZE*20-1:0]  data_out,
    output reg                        done
);

    // Decide interface mode at elaboration time
    localparam USE_BRAM = (INPUT_SIZE > 64) ? 1 : 0;

    //--------------------------------------------------------------------------
    // Weight ROM  (block RAM inferred for FC1 weights: 12,544×8 ≈ 6 RAMB18E1)
    //--------------------------------------------------------------------------
    (* rom_style = "block" *) reg signed [7:0] weights [0:INPUT_SIZE*OUTPUT_SIZE-1];
    reg signed [7:0] biases [0:OUTPUT_SIZE-1];

    initial begin
        $readmemh(WEIGHT_MEM, weights);
        $readmemh(BIAS_MEM,   biases);
    end

    //--------------------------------------------------------------------------
    // Input unpack array (only used in packed/FC2 mode)
    //--------------------------------------------------------------------------
    reg signed [19:0] input_unpacked [0:INPUT_SIZE-1];

    // Generate: only elaborate the unpack fanout for packed mode (FC2).
    // For BRAM mode (FC1) input_unpacked is unused; forcing it to zero avoids
    // synthesising a 15,680-bit combinational network.
    generate
        if (USE_BRAM == 0) begin : gen_unpack
            integer iu;
            always @(*) begin
                for (iu = 0; iu < INPUT_SIZE; iu = iu + 1)
                    input_unpacked[iu] = data_in[iu*20 +: 20];
            end
        end else begin : gen_no_unpack
            integer iu;
            always @(*) begin
                for (iu = 0; iu < INPUT_SIZE; iu = iu + 1)
                    input_unpacked[iu] = 20'sd0;
            end
        end
    endgenerate

    //--------------------------------------------------------------------------
    // Internal state
    //--------------------------------------------------------------------------
    reg signed [19:0] input_value;      // registered BRAM read data
    reg signed [31:0] accumulator;
    reg signed [31:0] neuron_results [0:OUTPUT_SIZE-1];

    reg [15:0] input_idx;
    reg [7:0]  neuron_idx;
    reg [15:0] weight_base;             // latched at start of each neuron (BRAM mode)
    reg [15:0] weight_idx;             // running index for BRAM-mode MAC (not used in packed mode)

    //--------------------------------------------------------------------------
    // FSM
    //--------------------------------------------------------------------------
    localparam IDLE        = 3'd0;
    localparam READ_PREFETCH = 3'd1;   // issue first BRAM read (BRAM mode)
    localparam COMPUTE     = 3'd2;
    localparam BIAS_ADD    = 3'd3;
    localparam NEXT_NEURON = 3'd4;
    localparam DONE_STATE  = 3'd5;

    reg [2:0] state;
    integer   ii;

    always @(posedge clk) begin
        if (rst) begin
            state        <= IDLE;
            done         <= 1'b0;
            accumulator  <= 32'sd0;
            input_idx    <= 16'd0;
            neuron_idx   <= 8'd0;
            bram_rd_addr <= 10'd0;
            input_value  <= 20'sd0;
            weight_base  <= 16'd0;
            weight_idx   <= 16'd0;
            data_out     <= {OUTPUT_SIZE*20{1'b0}};
            for (ii = 0; ii < OUTPUT_SIZE; ii = ii + 1)
                neuron_results[ii] <= 32'sd0;

        end else begin
            case (state)

                //--------------------------------------------------------------
                IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        accumulator  <= 32'sd0;
                        input_idx    <= 16'd0;
                        neuron_idx   <= 8'd0;
                        weight_base  <= 16'd0;
                        weight_idx   <= 16'd0;
                        if (USE_BRAM) begin
                            bram_rd_addr <= 10'd0;
                            state        <= READ_PREFETCH;
                        end else begin
                            state <= COMPUTE;
                        end
                    end
                end

                //--------------------------------------------------------------
                // Issue first two BRAM reads so data is available in COMPUTE.
                READ_PREFETCH: begin
                    bram_rd_addr <= 10'd1;
                    input_idx    <= 16'd1;
                    weight_base  <= neuron_idx * INPUT_SIZE;
                    weight_idx   <= neuron_idx * INPUT_SIZE;
                    state        <= COMPUTE;
                end

                //--------------------------------------------------------------
                COMPUTE: begin
                    if (USE_BRAM) begin
                        // ---- BRAM mode (FC1) ----
                        // Input value from PREVIOUS cycle's BRAM read (1-cycle latency)
                        input_value <= bram_rd_data;

                        if (input_idx > 16'd0) begin
                            // MAC: input from previous read × weight at running index
                            accumulator <= accumulator +
                                           ($signed(input_value) *
                                            $signed(weights[weight_idx]));
                            weight_idx  <= weight_idx + 16'd1;
                        end

                        if (input_idx == INPUT_SIZE[15:0]) begin
                            state     <= BIAS_ADD;
                            input_idx <= 16'd0;
                        end else begin
                            if (input_idx < (INPUT_SIZE[15:0] - 16'd1))
                                bram_rd_addr <= input_idx[9:0] + 10'd1;
                            input_idx <= input_idx + 16'd1;
                        end

                    end else begin
                        // ---- Packed mode (FC2) ----
                        // FIX: compute weight address COMBINATIONALLY using weight_base
                        // (which is registered and stable for the whole neuron).
                        // Old code used `weights[weight_idx]` where weight_idx was a
                        // non-blocking register lagging by 1 cycle → wrong weights.
                        accumulator <= accumulator +
                                       ($signed(input_unpacked[input_idx]) *
                                        $signed(weights[weight_base + input_idx]));

                        if (input_idx == (INPUT_SIZE[15:0] - 16'd1)) begin
                            state     <= BIAS_ADD;
                            input_idx <= 16'd0;
                        end else begin
                            input_idx <= input_idx + 16'd1;
                        end
                    end
                end

                //--------------------------------------------------------------
                BIAS_ADD: begin
                    accumulator <= accumulator +
                                   $signed({{24{biases[neuron_idx][7]}}, biases[neuron_idx]});
                    state <= NEXT_NEURON;
                end

                //--------------------------------------------------------------
                NEXT_NEURON: begin
                    // Apply ReLU (FC1) or pass raw logit (FC2)
                    if (APPLY_RELU && accumulator[31])
                        neuron_results[neuron_idx] <= 32'sd0;
                    else
                        neuron_results[neuron_idx] <= accumulator;

                    accumulator <= 32'sd0;

                    if (neuron_idx == (OUTPUT_SIZE[7:0] - 8'd1)) begin
                        state <= DONE_STATE;
                    end else begin
                        neuron_idx  <= neuron_idx + 8'd1;
                        // Latch weight base for the NEXT neuron before re-entering COMPUTE
                        weight_base <= (neuron_idx + 8'd1) * INPUT_SIZE;
                        weight_idx  <= (neuron_idx + 8'd1) * INPUT_SIZE;

                        if (USE_BRAM) begin
                            bram_rd_addr <= 10'd0;
                            state        <= READ_PREFETCH;
                        end else begin
                            state <= COMPUTE;
                        end
                    end
                end

                //--------------------------------------------------------------
                DONE_STATE: begin
                    for (ii = 0; ii < OUTPUT_SIZE; ii = ii + 1)
                        data_out[ii*20 +: 20] <= neuron_results[ii][19:0];
                    done  <= 1'b1;
                    state <= IDLE;
                end

                default: state <= IDLE;

            endcase
        end
    end

endmodule
