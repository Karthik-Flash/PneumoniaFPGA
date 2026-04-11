`timescale 1ns / 1ps
//==============================================================================
// line_buffer.v  -- V3 Bug-Fixed Version
//
// BUG FIXED (this file):
//   Middle row (row1) condition was `row_wr < 2`, forcing all three middle-row
//   pixels to ZERO for the first two rows of input (row_wr = 0 and row_wr = 1).
//
//   How the line buffer works (center convention):
//     row2 = current row being written (pixel_in at col_wr = bottom of window)
//     row1 = one row ago              = center row of 3x3 window
//     row0 = two rows ago             = top row of 3x3 window
//   → the output window corresponds to center at image row (row_wr - 1).
//
//   Correct zero-padding for PyTorch padding=1:
//     Top row  (row0): zero when row_wr < 1  (row -2 doesn't exist)
//     Middle   (row1): zero when row_wr < 1  (row -1 doesn't exist, center is row -1 only when row_wr=0)
//     Bottom   (row2): never forced to zero (pixel_in is center pixel of current live row)
//
//   With the old `row_wr < 2` condition the middle row was zeroed even when
//   row_wr = 1 (center = image row 0). row1 held real row-0 data but the
//   comparison zeroed it out, corrupting all 28 convolution windows for row 0.
//
//   FIX: change three `row_wr < 2` guards to `row_wr < 1`.
//
// Also retained from previous version:
//   Bottom-right pixel (row2[col_wr+1]) fix — both branches now correctly
//   return 8'h00 for border and row2[col_wr+1] for interior.
//==============================================================================

module line_buffer #(
    parameter IMG_WIDTH = 28
)(
    input  wire       clk,
    input  wire       rst,
    input  wire       start,
    input  wire [7:0] pixel_in,
    input  wire       pixel_valid,
    output reg [71:0] window_out,
    output reg        window_valid
);
    reg [7:0] row0 [0:IMG_WIDTH-1];
    reg [7:0] row1 [0:IMG_WIDTH-1];
    reg [7:0] row2 [0:IMG_WIDTH-1];

    reg [4:0] col_wr;
    reg [5:0] row_wr;
    reg [9:0] pixel_count;

    integer m;

    always @(posedge clk) begin
        if (rst || start) begin
            col_wr       <= 0;
            row_wr       <= 0;
            pixel_count  <= 0;
            window_out   <= 72'h0;
            window_valid <= 0;
            for (m = 0; m < IMG_WIDTH; m = m + 1) begin
                row0[m] <= 8'h00;
                row1[m] <= 8'h00;
                row2[m] <= 8'h00;
            end

        end else if (!pixel_valid) begin
            window_valid <= 0;

        end else begin   // pixel_valid

            // Write incoming pixel into the current row buffer
            row2[col_wr] <= pixel_in;

            // Assemble 3x3 window:
            //   row0 = two rows ago (top of window for center at row_wr-1)
            //   row1 = one row ago  (middle/center row)
            //   row2 = current row  (bottom of window), pixel_in = bottom-center
            //
            // Zero-padding rules (match PyTorch Conv2d padding=1):
            //   Top  row: zero when row_wr < 1  (image row above center is -2, doesn't exist)
            //   Mid  row: zero when row_wr < 1  (FIX: was row_wr < 2 — wrongly zeroed row 0 center)
            //   Left col: zero when col_wr == 0
            //   Right col: zero when col_wr == IMG_WIDTH-1
            window_out <= {
                // --- Top row (row0 = image row row_wr-2) ---
                (row_wr < 1 || col_wr == 0)         ? 8'h00 : row0[col_wr-1],
                (row_wr < 1)                         ? 8'h00 : row0[col_wr],
                (row_wr < 1 || col_wr==IMG_WIDTH-1) ? 8'h00 : row0[col_wr+1],
                // --- Middle row (row1 = image row row_wr-1 = center row) ---
                // FIX: was `row_wr < 2` — zeroed row1 even at row_wr=1 when row1 had real data
                (row_wr < 1 || col_wr == 0)         ? 8'h00 : row1[col_wr-1],
                (row_wr < 1)                         ? 8'h00 : row1[col_wr],
                (row_wr < 1 || col_wr==IMG_WIDTH-1) ? 8'h00 : row1[col_wr+1],
                // --- Bottom row (row2 = current image row row_wr) ---
                (col_wr == 0)                        ? 8'h00 : row2[col_wr-1],
                pixel_in,                                        // bottom-center = current pixel
                (col_wr == IMG_WIDTH-1)              ? 8'h00 : row2[col_wr+1]
            };

            window_valid <= pixel_valid;

            // Advance column; shift rows at end of each row
            if (col_wr == IMG_WIDTH-1) begin
                col_wr <= 0;
                row_wr <= row_wr + 1;
                for (m = 0; m < IMG_WIDTH; m = m + 1) begin
                    row0[m] <= row1[m];
                    row1[m] <= row2[m];
                end
            end else begin
                col_wr <= col_wr + 1;
            end

            pixel_count <= pixel_count + 1;
        end
    end

endmodule
