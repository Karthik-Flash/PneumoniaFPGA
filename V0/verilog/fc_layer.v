`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Module Name: fc_layer
// Description: Parameterized Fully Connected Layer with ROM-based Weights
//              for TinyML Cancer Detection Accelerator
//
// Port Connections:
//   - clk           : System clock (100 MHz)
//   - rst           : Synchronous active-high reset
//   - data_in       : Packed input vector [INPUT_SIZE*20-1:0]
//   - start         : Trigger signal to begin computation
//   - data_out      : Packed output vector [OUTPUT_SIZE*20-1:0]
//   - done          : Pulses high when computation is complete
//
// Parameters:
//   - INPUT_SIZE    : Number of input neurons (784 for FC1, 16 for FC2)
//   - OUTPUT_SIZE   : Number of output neurons (16 for FC1, 2 for FC2)
//   - WEIGHT_MEM    : Path to weight .mem file
//   - BIAS_MEM      : Path to bias .mem file
//
// Operation:
//   For each output neuron j:
//     1. Compute: sum = Σ(input[i] * weight[j][i]) for i=0 to INPUT_SIZE-1
//     2. Add bias: sum = sum + bias[j]
//     3. Apply ReLU: output[j] = max(0, sum)
//
// Memory Layout:
//   - Weights: neuron0_w0...neuron0_wN, neuron1_w0...neuron1_wN, ...
//   - Biases: bias[0], bias[1], ..., bias[OUTPUT_SIZE-1]
//
// Timing:
//   - Latency: INPUT_SIZE * OUTPUT_SIZE + OUTPUT_SIZE clock cycles
//   - FC1: 784 * 16 + 16 = 12,560 cycles
//   - FC2: 16 * 2 + 2 = 34 cycles
//
// FSM States:
//   - IDLE: Waiting for start signal
//   - COMPUTE: MAC operations for current neuron
//   - BIAS_ADD: Add bias and apply ReLU
//   - NEXT_NEURON: Move to next output neuron
//   - DONE: Assert done signal
//
//////////////////////////////////////////////////////////////////////////////////

module fc_layer #(
    parameter INPUT_SIZE = 784,
    parameter OUTPUT_SIZE = 16,
    parameter WEIGHT_MEM = "fc1_weights.mem",
    parameter BIAS_MEM = "fc1_bias.mem",
    parameter APPLY_RELU = 1  // 1 for FC1 (apply ReLU), 0 for FC2 (raw logits)
)(
    input wire clk,
    input wire rst,
    input wire [INPUT_SIZE*20-1:0] data_in,  // Packed input vector
    input wire start,                         // Start computation
    output reg [OUTPUT_SIZE*20-1:0] data_out, // Packed output vector
    output reg done                           // Done flag
);

    // ROM arrays for weights and biases
    reg signed [7:0] weights [0:INPUT_SIZE*OUTPUT_SIZE-1];  // Weight ROM
    reg signed [7:0] biases [0:OUTPUT_SIZE-1];              // Bias ROM
    
    // Internal signals
    reg signed [19:0] input_unpacked [0:INPUT_SIZE-1];      // Unpacked inputs
    reg signed [31:0] accumulator;
    reg signed [31:0] neuron_results [0:OUTPUT_SIZE-1];     // Store results
    
    // Counters and indices
    reg [15:0] input_idx;      // Current input index (0 to INPUT_SIZE-1)
    reg [7:0] neuron_idx;      // Current neuron index (0 to OUTPUT_SIZE-1)
    
    // FSM states
    localparam IDLE = 3'd0;
    localparam COMPUTE = 3'd1;
    localparam BIAS_ADD = 3'd2;
    localparam NEXT_NEURON = 3'd3;
    localparam DONE_STATE = 3'd4;
    
    reg [2:0] state;
    
    integer i;
    
    // Load weights and biases from .mem files
    initial begin
        $readmemh(WEIGHT_MEM, weights);
        $readmemh(BIAS_MEM, biases);
    end
    
    // Unpack input vector into array for easier indexing
    always @(*) begin
        for (i = 0; i < INPUT_SIZE; i = i + 1) begin
            input_unpacked[i] = data_in[i*20 +: 20];
        end
    end
    
    // Main FSM
    always @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            done <= 1'b0;
            accumulator <= 24'sd0;
            input_idx <= 16'd0;
            neuron_idx <= 8'd0;
            data_out <= {OUTPUT_SIZE*20{1'b0}};
            
            for (i = 0; i < OUTPUT_SIZE; i = i + 1) begin
                neuron_results[i] <= 24'sd0;
            end
            
        end else begin
            case (state)
                
                IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        state <= COMPUTE;
                        accumulator <= 24'sd0;
                        input_idx <= 16'd0;
                        neuron_idx <= 8'd0;
                    end
                end
                
                COMPUTE: begin
                    // Perform MAC: accumulator += input[input_idx] * weight[neuron_idx][input_idx]
                    // Weight index: neuron_idx * INPUT_SIZE + input_idx
                    accumulator <= accumulator + 
                                   ($signed(input_unpacked[input_idx]) * 
                                    $signed(weights[neuron_idx * INPUT_SIZE + input_idx]));
                    
                    if (input_idx == INPUT_SIZE - 1) begin
                        // Finished all inputs for this neuron
                        state <= BIAS_ADD;
                        input_idx <= 16'd0;
                    end else begin
                        input_idx <= input_idx + 1;
                    end
                end
                
                BIAS_ADD: begin
                    // Add bias and apply ReLU
                    // Extend bias from 8-bit to 24-bit
                    accumulator <= accumulator + $signed({{16{biases[neuron_idx][7]}}, biases[neuron_idx]});
                    state <= NEXT_NEURON;
                end
                
                NEXT_NEURON: begin
                    // Apply ReLU (FC1 only) or pass through raw logits (FC2)
                    if (APPLY_RELU && (accumulator[31] == 1'b1)) begin
                        // ReLU: Negative values clamp to zero
                        neuron_results[neuron_idx] <= 24'sd0;
                    end else begin
                        // Keep value (positive for FC1, or any value for FC2)
                        neuron_results[neuron_idx] <= accumulator;
                    end
                    
                    // Reset accumulator for next neuron
                    accumulator <= 24'sd0;
                    
                    if (neuron_idx == OUTPUT_SIZE - 1) begin
                        // Finished all neurons
                        state <= DONE_STATE;
                    end else begin
                        // Move to next neuron
                        neuron_idx <= neuron_idx + 1;
                        state <= COMPUTE;
                    end
                end
                
                DONE_STATE: begin
                    // Pack results into output vector
                    for (i = 0; i < OUTPUT_SIZE; i = i + 1) begin
                        data_out[i*20 +: 20] <= neuron_results[i][31:12];  // Truncate to 20 bits
                    end
                    done <= 1'b1;
                    state <= IDLE;
                end
                
                default: begin
                    state <= IDLE;
                end
                
            endcase
        end
    end

endmodule
