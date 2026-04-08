`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Module Name: top_accelerator
// Description: Top-level TinyML Pneumonia Detection FPGA Accelerator
//              Integrates complete inference pipeline
//
// Port Connections:
//   - clk             : System clock (100 MHz)
//   - rst             : Synchronous active-high reset
//   - start           : Begin inference on new image
//   - pixel_in        : 8-bit unsigned pixel stream (0-255)
//   - pixel_valid     : High when pixel_in is valid
//   - cancer_detected : 1-bit output (HIGH = pneumonia detected)
//   - result_valid    : High when cancer_detected is valid
//
// Complete Data Flow:
//   1. Pixel Stream (784 pixels)
//   2. Line Buffer with padding=1 -> 3x3 windows (784 windows)
//   3. MAC Array (4 filters) -> Convolution (28x28 feature maps)
//   4. ReLU -> Non-linearity
//   5. Max Pool (2x2) -> Downsample to 14x14
//   6. FC1 (784->16) -> Feature extraction
//   7. FC2 (16->2) -> Classification logits
//   8. Argmax -> cancer_detected = (logit[1] > logit[0])
//
// Timing:
//   - Total latency: ~14,000 cycles @ 100 MHz
//   - Throughput: ~7,000 inferences/second
//
// Architecture Notes:
//   - 4 parallel MAC units for 4 conv filters
//   - Sequential processing for FC layers (resource efficient)
//   - Single clock domain, fully synchronous
//   - Zero-padding in line buffer matches PyTorch Conv2d(padding=1)
//
//////////////////////////////////////////////////////////////////////////////////

module top_accelerator(
    input wire clk,
    input wire rst,
    input wire start,
    input wire [7:0] pixel_in,
    input wire pixel_valid,
    output wire cancer_detected,
    output wire result_valid
);

    // ========== ROM for Convolution Weights and Bias ==========
    
    // Conv1 weights: 4 filters × 9 weights = 36 values (signed 8-bit)
    reg signed [7:0] conv_weights [0:35];
    // Conv1 bias: 4 values (signed 8-bit)
    reg signed [7:0] conv_bias_rom [0:3];
    
    initial begin
        $readmemh("conv1_weights.mem", conv_weights);
        $readmemh("conv1_bias.mem", conv_bias_rom);
    end
    
    // ========== Internal Signals ==========
    
    // Line buffer signals
    wire [71:0] window_out;
    wire window_valid;
    
    // MAC unit signals (4 parallel units for 4 filters)
    wire signed [19:0] mac_out [0:3];
    wire mac_valid [0:3];
    
    // ReLU outputs (4 channels)
    wire [19:0] relu_out [0:3];
    
    // Convolution feature map storage (4 channels x 28x28)
    // After conv+ReLU with padding=1, produces 28x28 output
    reg [19:0] feature_maps [0:3][0:27][0:27];
    reg [9:0] fm_write_count;  // Counts valid conv outputs (0-783 for 28x28)
    reg conv_complete;
    reg [4:0] fm_row, fm_col;  // Track row/col position
    
    // Max pooling signals (4 channels x 14x14)
    // 28x28 with 2x2 pooling produces 14x14 output
    reg [19:0] pooled_maps [0:3][0:13][0:13];
    wire signed [19:0] pool_out [0:3];
    reg [7:0] pool_count;
    reg pool_complete;
    reg [4:0] pool_row, pool_col;  // Track pooling position
    
    // Flattened pooled data for FC1 input (784 values = 4x14x14)
    wire [15679:0] fc1_input;   // 784 x 20 bits
    wire [319:0] fc1_output;    // 16 x 20 bits
    wire fc1_done;
    
    // FC2 signals
    wire [39:0] fc2_output;     // 2 x 20 bits (2 logits)
    wire fc2_done;
    
    // FSM control signals
    wire [2:0] fsm_state;
    wire conv_en, pool_en, fc1_en, fc2_en;
    wire output_valid;
    
    // Final classification
    wire signed [19:0] logit0, logit1;
    reg cancer_flag;
    
    integer i, j, k;
    
    // ========== Module Instantiations ==========
    
    // Line Buffer: Generates 3x3 sliding windows with zero-padding
line_buffer #(
    .IMG_WIDTH(28)
) u_line_buffer (
    .clk(clk),
    .rst(rst),
    .start(start),
    .pixel_in(pixel_in),
    .pixel_valid(pixel_valid && conv_en),
    .window_out(window_out),
    .window_valid(window_valid)
);
    // MAC Units: 4 parallel units (one per conv filter)
    // Real convolution using trained weights from ROM
    generate
        genvar filter_idx;
        for (filter_idx = 0; filter_idx < 4; filter_idx = filter_idx + 1) begin : mac_array
            // Compute weighted sum: MAC = sum(pixel[i] * weight[f*9+i]) + bias[f]
            assign mac_out[filter_idx] = 
                ($signed({1'b0, window_out[71:64]}) * conv_weights[filter_idx*9+0]) +
                ($signed({1'b0, window_out[63:56]}) * conv_weights[filter_idx*9+1]) +
                ($signed({1'b0, window_out[55:48]}) * conv_weights[filter_idx*9+2]) +
                ($signed({1'b0, window_out[47:40]}) * conv_weights[filter_idx*9+3]) +
                ($signed({1'b0, window_out[39:32]}) * conv_weights[filter_idx*9+4]) +
                ($signed({1'b0, window_out[31:24]}) * conv_weights[filter_idx*9+5]) +
                ($signed({1'b0, window_out[23:16]}) * conv_weights[filter_idx*9+6]) +
                ($signed({1'b0, window_out[15:8]})  * conv_weights[filter_idx*9+7]) +
                ($signed({1'b0, window_out[7:0]})   * conv_weights[filter_idx*9+8]) +
                {{12{conv_bias_rom[filter_idx][7]}}, conv_bias_rom[filter_idx]};
            
            assign mac_valid[filter_idx] = window_valid;
        end
    endgenerate
    
    // ReLU units: 4 parallel (one per filter)
    generate
        genvar relu_idx;
        for (relu_idx = 0; relu_idx < 4; relu_idx = relu_idx + 1) begin : relu_array
            relu u_relu (
                .data_in(mac_out[relu_idx]),
                .data_out(relu_out[relu_idx])
            );
        end
    endgenerate
    
    // Max Pooling units: 4 parallel (one per channel)
    // Connect 2x2 windows from feature maps
    generate
        genvar pool_idx;
        for (pool_idx = 0; pool_idx < 4; pool_idx = pool_idx + 1) begin : pool_array
            max_pool u_max_pool (
                .in0(feature_maps[pool_idx][pool_row*2][pool_col*2]),
                .in1(feature_maps[pool_idx][pool_row*2][pool_col*2+1]),
                .in2(feature_maps[pool_idx][pool_row*2+1][pool_col*2]),
                .in3(feature_maps[pool_idx][pool_row*2+1][pool_col*2+1]),
                .max_out(pool_out[pool_idx])
            );
        end
    endgenerate
    
    // FC1 Layer: 784 inputs -> 16 outputs (with ReLU activation)
    fc_layer #(
        .INPUT_SIZE(784),
        .OUTPUT_SIZE(16),
        .WEIGHT_MEM("fc1_weights.mem"),
        .BIAS_MEM("fc1_bias.mem"),
        .APPLY_RELU(1)  // Apply ReLU for FC1
    ) u_fc1 (
        .clk(clk),
        .rst(rst),
        .data_in(fc1_input),
        .start(fc1_en),
        .data_out(fc1_output),
        .done(fc1_done)
    );
    
    // FC2 Layer: 16 inputs -> 2 outputs (NO ReLU - raw logits for argmax)
    fc_layer #(
        .INPUT_SIZE(16),
        .OUTPUT_SIZE(2),
        .WEIGHT_MEM("fc2_weights.mem"),
        .BIAS_MEM("fc2_bias.mem"),
        .APPLY_RELU(0)  // No ReLU for FC2 - need raw logits
    ) u_fc2 (
        .clk(clk),
        .rst(rst),
        .data_in(fc1_output),
        .start(fc2_en),
        .data_out(fc2_output),
        .done(fc2_done)
    );
    
    // FSM Control: Master state machine
    fsm_control u_fsm (
        .clk(clk),
        .rst(rst),
        .start(start),
        .conv_done(conv_complete),
        .pool_done(pool_complete),
        .fc1_done(fc1_done),
        .fc2_done(fc2_done),
        .state(fsm_state),
        .conv_en(conv_en),
        .pool_en(pool_en),
        .fc1_en(fc1_en),
        .fc2_en(fc2_en),
        .output_valid(output_valid)
    );
    
    // ========== Convolution + ReLU Feature Map Storage ==========
    
    always @(posedge clk) begin
        if (rst || start) begin
            fm_write_count <= 10'd0;
            conv_complete <= 1'b0;
            fm_row <= 5'd0;
            fm_col <= 5'd0;
            // Clear feature maps (28x28 for each of 4 channels)
            for (i = 0; i < 4; i = i + 1) begin
                for (j = 0; j < 28; j = j + 1) begin
                    for (k = 0; k < 28; k = k + 1) begin
                        feature_maps[i][j][k] <= 20'd0;
                    end
                end
            end
        end else if (conv_en && window_valid) begin
            // Store ReLU outputs into feature maps (28x28 = 784 windows)
            for (i = 0; i < 4; i = i + 1) begin
                feature_maps[i][fm_row][fm_col] <= relu_out[i];
            end
            
            // Update row/col position
            if (fm_col == 5'd27) begin
                fm_col <= 5'd0;
                fm_row <= fm_row + 1;
            end else begin
                fm_col <= fm_col + 1;
            end
            
            // Check if we've processed all 784 windows (28x28)
            if (fm_write_count == 10'd783) begin
                conv_complete <= 1'b1;
                fm_write_count <= 10'd0;
            end else begin
                fm_write_count <= fm_write_count + 1;
            end
        end 
    end
    
    // ========== Max Pooling Logic ==========
    
    always @(posedge clk) begin
        if (rst || start) begin
    pool_count    <= 8'd0;
    pool_complete <= 1'b0;
    pool_row      <= 5'd0;
    pool_col      <= 5'd0;
    for (i = 0; i < 4; i = i + 1) begin
        for (j = 0; j < 14; j = j + 1) begin
            for (k = 0; k < 14; k = k + 1) begin
                pooled_maps[i][j][k] <= 20'd0;
            end
        end
    end
end else if (pool_en) begin
            // Perform 2x2 pooling on each 28x28 feature map
            // Output: 4 channels x 14x14 = 784 values
            // pool_out is combinatorially computed by max_pool instances
            
            // Store pooling results
            for (i = 0; i < 4; i = i + 1) begin
                pooled_maps[i][pool_row][pool_col] <= pool_out[i];
            end
            
            // Update pooling position
            if (pool_col == 5'd13) begin
                pool_col <= 5'd0;
                if (pool_row == 5'd13) begin
                    pool_complete <= 1'b1;
                    pool_row <= 5'd0;
                    pool_count <= 8'd0;
                end else begin
                    pool_row <= pool_row + 1;
                    pool_count <= pool_count + 1;
                end
            end else begin
                pool_col <= pool_col + 1;
                pool_count <= pool_count + 1;
            end
        end 
    end
    
    // ========== FC1 Input Packing ==========
    
    // Pack pooled feature maps into flat vector for FC1 (784 values = 4x14x14)
    generate
        genvar fc1_idx;
        for (fc1_idx = 0; fc1_idx < 784; fc1_idx = fc1_idx + 1) begin : fc1_input_pack
            // Map 1D index to (channel, row, col)
            // channel = fc1_idx / 196, row = (fc1_idx % 196) / 14, col = fc1_idx % 14
            assign fc1_input[fc1_idx*20 +: 20] = pooled_maps[fc1_idx/196][(fc1_idx%196)/14][fc1_idx%14];
        end
    endgenerate
    
    // ========== Argmax and Final Classification ==========
    
    // Extract 2 logits from FC2 output
    assign logit0 = fc2_output[19:0];   // Normal score
    assign logit1 = fc2_output[39:20];  // Pneumonia score
    
    
    // Argmax comparator: if logit1 > logit0, predict pneumonia
    always @(posedge clk) begin
        if (rst || start) begin
            cancer_flag <= 1'b0;
        end else if (output_valid) begin
            cancer_flag <= (logit1 > logit0) ? 1'b1 : 1'b0;
        end
    end
    
    // ========== Output Assignment ==========
    
    assign cancer_detected = cancer_flag;
    assign result_valid = output_valid;
always @(posedge clk) begin
    if (output_valid) begin
        $display("LOGITS: logit0(normal)=%0d logit1(pneumonia)=%0d -> cancer_detected=%b",
                 $signed(logit0), $signed(logit1), cancer_flag);
    end
end
endmodule
