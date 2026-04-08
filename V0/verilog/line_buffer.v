`timescale 1ns / 1ps
module line_buffer #(
    parameter IMG_WIDTH = 28
)(
    input wire clk,
    input wire rst,
    input wire start,
    input wire [7:0] pixel_in,
    input wire pixel_valid,
    output reg [71:0] window_out,
    output reg window_valid
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
    for (m = 0; m < IMG_WIDTH; m = m+1) begin
        row0[m] <= 8'h00;
        row1[m] <= 8'h00;
        row2[m] <= 8'h00;
    end
    end else if (!pixel_valid && !rst) begin
        // clear row buffers between images when idle
        window_valid <= 0;
        end else if (pixel_valid) begin

            // Write pixel into row2 (current row)
            row2[col_wr] <= pixel_in;

            // Build window using row0=top, row1=mid, row2=bottom
            // Center pixel is row1[col_wr] which was written last row
            window_out <= {
                // Top row (row0)
                (row_wr < 1 || col_wr == 0)              ? 8'h00 : row0[col_wr-1],
                (row_wr < 1)                              ? 8'h00 : row0[col_wr],
                (row_wr < 1 || col_wr==IMG_WIDTH-1)      ? 8'h00 : row0[col_wr+1],
                // Middle row (row1)
                (row_wr < 2 || col_wr == 0)              ? 8'h00 : row1[col_wr-1],
                (row_wr < 2)                              ? 8'h00 : row1[col_wr],
                (row_wr < 2 || col_wr==IMG_WIDTH-1)      ? 8'h00 : row1[col_wr+1],
                // Bottom row (row2 = current)
                (col_wr == 0)                             ? 8'h00 : row2[col_wr-1],
                pixel_in,
                (col_wr == IMG_WIDTH-1)                  ? 8'h00 : 8'h00
            };

            // Valid only after 2 full rows have been stored
            window_valid <= pixel_valid;

            // Advance column
            if (col_wr == IMG_WIDTH-1) begin
                col_wr <= 0;
                row_wr <= row_wr + 1;
                // Shift rows up
                for (m = 0; m < IMG_WIDTH; m = m+1) begin
                    row0[m] <= row1[m];
                    row1[m] <= row2[m];
                end
            end else begin
                col_wr <= col_wr + 1;
            end

            pixel_count <= pixel_count + 1;
        end else begin
            window_valid <= 0;
        end
    end
endmodule