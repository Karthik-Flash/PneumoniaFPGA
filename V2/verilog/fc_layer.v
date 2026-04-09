`timescale 1ns / 1ps
//==============================================================================
// fc_layer.v  -- V3 Bug-Fixed Version
//
// BUG FIXED (this file):
//   READ_PREFETCH set input_idx = 16'd1, SKIPPING the first warmup cycle.
//   This caused:
//     - Cycle 1 of COMPUTE: input_value still = 0 (not yet latched) → accumulates 0×w[0]
//     - Cycle 2 of COMPUTE: input_value = pixel[0] but weight_idx = 1 → pixel[0]×w[1]
//     - Pattern: pixel[k] × w[k+1] for all k → every neuron completely wrong
//     - Last pixel pixel[783] × w[783] never computed at all
//   FIX: set input_idx = 16'd0 in READ_PREFETCH.
//     - Cycle 1 of COMPUTE (idx=0): input_value latched from bram_rd_data, NO accumulate
//     - Cycle 2 onward (idx=1..784): pixel[k-1] × w[k-1] ← correct alignment
//     - Costs 1 extra cycle per neuron (784+1 instead of 784) = negligible
//
// KEY FIXES retained from V3:
//   1. DATA_IN_WIDTH parameter — FC1 port trivially small (20 bits), avoids 15,680-bit fanout
//   2. FC2 weight index computed inline (weight_base + input_idx) — no off-by-one lag
//   3. input_unpacked fanout guarded by generate (USE_BRAM==0 only)
//
// Timing:
//   FC1 (BRAM mode,  784 inputs, 16 neurons):  (784+2)*16 = 12,576 + 16 = ~12,592 cycles
//   FC2 (packed mode, 16 inputs,  2 neurons):  16*2 + overhead ≈ 36 cycles
//==============================================================================

module fc_layer #(
    parameter INPUT_SIZE    = 784,
    parameter OUTPUT_SIZE   = 16,
    parameter DATA_IN_WIDTH = 20,
    parameter WEIGHT_MEM    = "fc1_weights.mem",
    parameter BIAS_MEM      = "fc1_bias.mem",
    parameter APPLY_RELU    = 1
)(
    input  wire                       clk,
    input  wire                       rst,
    output reg  [9:0]                 bram_rd_addr,
    input  wire [19:0]                bram_rd_data,
    input  wire [DATA_IN_WIDTH-1:0]   data_in,
    input  wire                       start,
    output reg  [OUTPUT_SIZE*20-1:0]  data_out,
    output reg                        done
);

    localparam USE_BRAM = (INPUT_SIZE > 64) ? 1 : 0;

    //--------------------------------------------------------------------------
    // Weight ROM
    //--------------------------------------------------------------------------
    (* rom_style = "block" *) reg signed [7:0] weights [0:INPUT_SIZE*OUTPUT_SIZE-1];
    reg signed [7:0] biases [0:OUTPUT_SIZE-1];

    initial begin
        $readmemh(WEIGHT_MEM, weights);
        $readmemh(BIAS_MEM,   biases);
    end

    //--------------------------------------------------------------------------
    // Input unpack (packed/FC2 mode only)
    //--------------------------------------------------------------------------
    reg signed [19:0] input_unpacked [0:INPUT_SIZE-1];

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
    reg signed [19:0] input_value;
    reg signed [31:0] accumulator;
    reg signed [31:0] neuron_results [0:OUTPUT_SIZE-1];

    reg [15:0] input_idx;
    reg [7:0]  neuron_idx;
    reg [15:0] weight_base;
    reg [15:0] weight_idx;

    //--------------------------------------------------------------------------
    // FSM
    //--------------------------------------------------------------------------
    localparam IDLE          = 3'd0;
    localparam READ_PREFETCH = 3'd1;
    localparam COMPUTE       = 3'd2;
    localparam BIAS_ADD      = 3'd3;
    localparam NEXT_NEURON   = 3'd4;
    localparam DONE_STATE    = 3'd5;

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
                // Issue bram_rd_addr=1 so pixel[1] is ready one cycle into COMPUTE.
                // FIX: input_idx stays at 0 (was 1) so the first COMPUTE cycle
                //      is a "warmup" latch: input_value <= pixel[0], no accumulate.
                //      COMPUTE cycle 2 (input_idx=1): pixel[0] x w[0] ✓
                //--------------------------------------------------------------
                READ_PREFETCH: begin
                    bram_rd_addr <= 10'd1;
                    input_idx    <= 16'd0;    // FIX: was 16'd1 — caused off-by-one misalignment
                    weight_base  <= neuron_idx * INPUT_SIZE;
                    weight_idx   <= neuron_idx * INPUT_SIZE;
                    state        <= COMPUTE;
                end

                //--------------------------------------------------------------
                COMPUTE: begin
                    if (USE_BRAM) begin
                        // Latch BRAM output (1-cycle latency)
                        input_value <= bram_rd_data;

                        // Skip accumulate on idx=0 (warmup: latching pixel[0])
                        if (input_idx > 16'd0) begin
                            accumulator <= accumulator +
                                           ($signed(input_value) *
                                            $signed(weights[weight_idx]));
                            weight_idx  <= weight_idx + 16'd1;
                        end

                        if (input_idx == INPUT_SIZE[15:0]) begin
                            // idx==INPUT_SIZE: last MAC was pixel[INPUT_SIZE-1] x w[last]
                            state     <= BIAS_ADD;
                            input_idx <= 16'd0;
                        end else begin
                            if (input_idx < (INPUT_SIZE[15:0] - 16'd1))
                                bram_rd_addr <= input_idx[9:0] + 10'd1;
                            input_idx <= input_idx + 16'd1;
                        end

                    end else begin
                        // Packed mode (FC2): inline weight address, no latency issue
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
                    if (APPLY_RELU && accumulator[31])
                        neuron_results[neuron_idx] <= 32'sd0;
                    else
                        neuron_results[neuron_idx] <= accumulator;

                    accumulator <= 32'sd0;

                    if (neuron_idx == (OUTPUT_SIZE[7:0] - 8'd1)) begin
                        state <= DONE_STATE;
                    end else begin
                        neuron_idx  <= neuron_idx + 8'd1;
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
