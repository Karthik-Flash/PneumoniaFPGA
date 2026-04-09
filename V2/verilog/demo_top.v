`timescale 1ns / 1ps
//==============================================================================
// demo_top.v  -- PYNQ-Z2 Hardware Demo Wrapper
//
// What this does ON THE ACTUAL BOARD:
//   After reset is released (BTN0 released), it automatically streams the
//   pre-loaded pneumonia test image through the accelerator once, then holds
//   the result on the LEDs until reset is pressed again.
//
// LED behaviour:
//   LD0 (R14) = cancer_detected  : ON = Pneumonia detected, OFF = Normal
//   LD1 (P14) = result_valid     : Pulses for ~1 sec after inference completes
//   LD2 (N16) = running          : ON while inference is in progress
//   LD3 (M14) = done_latch       : ON permanently once any result is available
//
// Controls:
//   BTN0 (D19) = rst  : Hold to reset, release to start a new inference
//   BTN1 (D20) = re-trigger: pulse to run inference again (without full reset)
//
// Notes:
//   - pixel_in data comes from image_pneumonia1.mem baked into Block RAM
//     at synthesis time via $readmemh. No external data source needed.
//   - The image streams automatically once rst is released; no button needed.
//   - The 125 MHz on-board clock is divided down to 80 MHz via a simple
//     counter divider (or tie clk directly to 125 MHz — the accelerator
//     was closed at 80 MHz so 125 MHz will NOT work; use the MMCM version
//     below or reduce to 62.5 MHz with a /2 divider for a safe demo).
//
// For a proper 80 MHz clock, instantiate a Clocking Wizard IP in your
// Vivado project: 125 MHz in -> 80 MHz out, then connect clk_80 below.
// This wrapper uses a simple /2 divider giving 62.5 MHz to stay safe.
//==============================================================================

module demo_top (
    input  wire       clk_125,      // 125 MHz on-board oscillator (H16)
    input  wire       btn0_rst,     // BTN0: active HIGH reset
    input  wire       btn1_retrig,  // BTN1: re-trigger inference
    output wire       ld0_pneumonia,// LED0: cancer_detected
    output wire       ld1_valid,    // LED1: result_valid (extended pulse)
    output wire       ld2_running,  // LED2: inference in progress
    output wire       ld3_done      // LED3: latched done
);

    //--------------------------------------------------------------------------
    // Clock divider: 125 MHz -> 62.5 MHz  (safe margin under 80 MHz budget)
    // To get exact 80 MHz, replace this with a Clocking Wizard IP.
    //--------------------------------------------------------------------------
    reg clk_div;
    always @(posedge clk_125) begin
        clk_div <= ~clk_div;
    end
    wire clk = clk_div;   // 62.5 MHz working clock

    //--------------------------------------------------------------------------
    // Synchronise buttons into the 62.5 MHz domain (2-FF synchroniser)
    //--------------------------------------------------------------------------
    reg btn0_s1, btn0_s2;
    reg btn1_s1, btn1_s2;
    always @(posedge clk) begin
        btn0_s1 <= btn0_rst;    btn0_s2 <= btn0_s1;
        btn1_s1 <= btn1_retrig; btn1_s2 <= btn1_s1;
    end
    wire rst   = btn0_s2;           // active high reset
    wire retrig_raw = btn1_s2;

    // Edge-detect BTN1 for re-trigger pulse
    reg retrig_prev;
    always @(posedge clk) retrig_prev <= retrig_raw;
    wire retrig_pulse = retrig_raw & ~retrig_prev;  // rising edge

    //--------------------------------------------------------------------------
    // Pre-loaded test image (784 bytes baked into BRAM at synthesis time)
    // Change the filename to test a normal image.
    //--------------------------------------------------------------------------
    reg [7:0] image_rom [0:783];
    initial $readmemh("image_pneumonia1.mem", image_rom);

    //--------------------------------------------------------------------------
    // Pixel streaming state machine
    //   WAIT  -> stream_START -> STREAM (784 pixels) -> DONE
    //--------------------------------------------------------------------------
    localparam ST_WAIT   = 2'd0;
    localparam ST_START  = 2'd1;
    localparam ST_STREAM = 2'd2;
    localparam ST_DONE   = 2'd3;

    reg [1:0]  stream_state;
    reg [9:0]  pix_idx;
    reg        start_reg;
    reg [7:0]  pixel_in_reg;
    reg        pixel_valid_reg;
    reg        done_latch;

    // Accelerator interface
    wire cancer_detected;
    wire result_valid;

    always @(posedge clk) begin
        if (rst) begin
            stream_state    <= ST_START;   // auto-start immediately after reset
            pix_idx         <= 10'd0;
            start_reg       <= 1'b0;
            pixel_in_reg    <= 8'h00;
            pixel_valid_reg <= 1'b0;
            done_latch      <= 1'b0;
        end else begin
            // Default de-assert
            start_reg       <= 1'b0;
            pixel_valid_reg <= 1'b0;

            // Re-trigger from BTN1 sends back to ST_START
            if (retrig_pulse && stream_state == ST_DONE)
                stream_state <= ST_START;

            case (stream_state)

                ST_WAIT: begin
                    // Idle — only re-trigger moves us forward
                end

                ST_START: begin
                    // Pulse start for 1 cycle, then stream
                    start_reg    <= 1'b1;
                    pix_idx      <= 10'd0;
                    stream_state <= ST_STREAM;
                end

                ST_STREAM: begin
                    pixel_in_reg    <= image_rom[pix_idx];
                    pixel_valid_reg <= 1'b1;
                    if (pix_idx == 10'd783) begin
                        pix_idx      <= 10'd0;
                        stream_state <= ST_DONE;
                    end else begin
                        pix_idx <= pix_idx + 10'd1;
                    end
                end

                ST_DONE: begin
                    // Wait for result; latch it
                    if (result_valid)
                        done_latch <= 1'b1;
                end

            endcase
        end
    end

    wire running = (stream_state == ST_STREAM);

    //--------------------------------------------------------------------------
    // Extend result_valid into a ~1-second visible pulse on LD1
    // 62.5 MHz * 62_500_000 cycles = 1.0 second
    //--------------------------------------------------------------------------
    reg [25:0] valid_stretch_cnt;
    reg        valid_stretched;

    always @(posedge clk) begin
        if (rst) begin
            valid_stretch_cnt <= 26'd0;
            valid_stretched   <= 1'b0;
        end else if (result_valid) begin
            valid_stretch_cnt <= 26'd62_500_000;
            valid_stretched   <= 1'b1;
        end else if (valid_stretch_cnt > 26'd0) begin
            valid_stretch_cnt <= valid_stretch_cnt - 26'd1;
        end else begin
            valid_stretched <= 1'b0;
        end
    end

    //--------------------------------------------------------------------------
    // Instantiate the accelerator
    //--------------------------------------------------------------------------
    top_accelerator u_accel (
        .clk           (clk),
        .rst           (rst),
        .start         (start_reg),
        .pixel_in      (pixel_in_reg),
        .pixel_valid   (pixel_valid_reg),
        .cancer_detected(cancer_detected),
        .result_valid  (result_valid)
    );

    //--------------------------------------------------------------------------
    // LED outputs
    //--------------------------------------------------------------------------
    assign ld0_pneumonia = cancer_detected;    // result pin
    assign ld1_valid     = valid_stretched;    // 1-sec blink on completion
    assign ld2_running   = running;            // on during the 784-cycle stream
    assign ld3_done      = done_latch;         // stays on after first result

endmodule
