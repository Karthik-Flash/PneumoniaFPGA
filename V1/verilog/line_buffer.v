`timescale 1ns / 1ps
//==============================================================================
// line_buffer.v  -- Bug-fixed Version
//
// FIX: Bottom-right pixel (row2[col_wr+1]) was ALWAYS 8'h00.
//   Old code:  (col_wr == IMG_WIDTH-1) ? 8'h00 : 8'h00
//   Both branches returned 0, so even interior pixels had zero right-neighbor.
//   Fix:       (col_wr == IMG_WIDTH-1) ? 8'h00 : row2[col_wr+1]
//   row2[col_wr+1] is the value written there in the PREVIOUS row (row2 is
//   shifted from row1 at end-of-row), which acts as the buffered right pixel.
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
            col_wr      <= 0;
            row_wr      <= 0;
            pixel_count <= 0;
            window_out  <= 72'h0;
            window_valid<= 0;
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
            //   row0 = two rows ago (top),  row1 = one row ago (middle),
            //   row2 = current row (bottom, partially filled)
            window_out <= {
                // --- Top row (row0) ---
                (row_wr < 1 || col_wr == 0)           ? 8'h00 : row0[col_wr-1],
                (row_wr < 1)                           ? 8'h00 : row0[col_wr],
                (row_wr < 1 || col_wr==IMG_WIDTH-1)   ? 8'h00 : row0[col_wr+1],
                // --- Middle row (row1) ---
                (row_wr < 2 || col_wr == 0)           ? 8'h00 : row1[col_wr-1],
                (row_wr < 2)                           ? 8'h00 : row1[col_wr],
                (row_wr < 2 || col_wr==IMG_WIDTH-1)   ? 8'h00 : row1[col_wr+1],
                // --- Bottom row (row2, current) ---
                (col_wr == 0)                          ? 8'h00 : row2[col_wr-1],
                pixel_in,                                               // center
                // FIX: was "(... ? 8'h00 : 8'h00)" — right pixel always zero!
                (col_wr == IMG_WIDTH-1)                ? 8'h00 : row2[col_wr+1]
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
