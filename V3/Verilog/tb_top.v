`timescale 1ns / 1ps

module tb_top;

    reg clk, rst, start;
    reg [7:0] pixel_in;
    reg pixel_valid;

    wire cancer_detected;
    wire result_valid;

    integer test_num, errors, pixel_idx;
    reg [7:0] image_mem [0:783];
    reg [2:0] prev_fsm_state;

    // Clock: 100 MHz
    initial begin clk = 0; forever #5 clk = ~clk; end

    top_accelerator dut (
        .clk(clk), .rst(rst), .start(start),
        .pixel_in(pixel_in), .pixel_valid(pixel_valid),
        .cancer_detected(cancer_detected), .result_valid(result_valid)
    );

    // Main sequence
    initial begin
        $display("========================================");
        $display("TinyML Pneumonia Detection Accelerator");
        $display("8-Image INT8-Verified Testbench");
        $display("========================================\n");

        rst = 1; start = 0; pixel_in = 8'h00;
        pixel_valid = 0; errors = 0; test_num = 0;
        prev_fsm_state = 3'bxxx;

        repeat(10) @(posedge clk); rst = 0;
        repeat(5)  @(posedge clk);
        $display("[%0t ns] Reset released, beginning tests\n", $time);

        // 4 pneumonia images (expect cancer_detected=1)
        run_test("image_pneumonia1.mem", 1, 1);
        run_test("image_pneumonia2.mem", 2, 1);
        run_test("image_pneumonia3.mem", 3, 1);
        run_test("image_pneumonia4.mem", 4, 1);

        // 4 normal images (expect cancer_detected=0)
        run_test("image_normal1.mem",   5, 0);
        run_test("image_normal2.mem",   6, 0);
        run_test("image_normal3.mem",   7, 0);
        run_test("image_normal4.mem",   8, 0);

        #100;
        $display("\n========================================");
        $display("TEST SUMMARY");
        $display("========================================");
        $display("Tests Passed: %0d/%0d", test_num - errors, test_num);
        $display("Errors:       %0d", errors);
        if (errors == 0)
            $display("\n*** ALL %0d TESTS PASSED ***", test_num);
        else
            $display("\n*** %0d TEST(S) FAILED ***", errors);
        $display("========================================\n");
        $finish;
    end

    // FSM state monitor (change-only)
    always @(posedge clk) begin
        if (dut.u_fsm.state !== prev_fsm_state) begin
            case (dut.u_fsm.state)
                3'b000: $display("[%0t ns] FSM -> IDLE",    $time);
                3'b001: $display("[%0t ns] FSM -> CONV",    $time);
                3'b010: $display("[%0t ns] FSM -> POOL",    $time);
                3'b011: $display("[%0t ns] FSM -> FC1",     $time);
                3'b100: $display("[%0t ns] FSM -> FC2",     $time);
                3'b101: $display("[%0t ns] FSM -> OUTPUT",  $time);
                default:$display("[%0t ns] FSM -> UNKNOWN", $time);
            endcase
            prev_fsm_state <= dut.u_fsm.state;
        end
    end

    // Global watchdog: 8 inferences × 250k cycles max = 2M cycles
    initial begin
        #400000000;
        $display("\n*** GLOBAL WATCHDOG TIMEOUT after 400ms ***");
        $finish;
    end

    // -------------------------------------------------------
    // Unified task: runs one image, checks against expected
    // -------------------------------------------------------
    task run_test;
        input [1000*8-1:0] filename;
        input integer       img_num;
        input integer       expected; // 1=pneumonia, 0=normal
        integer timeout_counter;
        begin
            test_num = test_num + 1;
            repeat(50) @(posedge clk);

            $display("========================================");
            if (expected == 1)
                $display("[Test %0d] Pneumonia Image %0d", test_num, img_num);
            else
                $display("[Test %0d] Normal Image %0d",   test_num, img_num - 4);
            $readmemh(filename, image_mem);
            $display("  Loaded: %0s", filename);
            $display("  DEBUG: First 5 pixels: %d %d %d %d %d", 
                     image_mem[0], image_mem[1], image_mem[2], image_mem[3], image_mem[4]);

            @(posedge clk); start = 1;
            @(posedge clk); start = 0;

            for (pixel_idx = 0; pixel_idx < 784; pixel_idx = pixel_idx + 1) begin
                @(posedge clk);
                pixel_in    = image_mem[pixel_idx];
                pixel_valid = 1;
            end
            @(posedge clk);
            pixel_valid = 0; pixel_in = 8'h00;
            $display("  Streamed 784 pixels. Waiting for result...");

            timeout_counter = 0;
            while (result_valid !== 1'b1 && timeout_counter < 2000000) begin
                @(posedge clk);
                timeout_counter = timeout_counter + 1;
            end

            if (timeout_counter >= 500000) begin
                $display("*** TIMEOUT on Test %0d ***", test_num);
                $finish;
            end

            repeat(2) @(posedge clk);

            if (cancer_detected === expected[0:0]) begin
                if (expected == 1)
                    $display("  [PASS] cancer_detected=1 (pneumonia correct)");
                else
                    $display("  [PASS] cancer_detected=0 (normal correct)");
            end else begin
                if (expected == 1)
                    $display("  [FAIL] cancer_detected=0 (missed pneumonia!)");
                else
                    $display("  [FAIL] cancer_detected=1 (false positive!)");
                errors = errors + 1;
            end
            $display("========================================\n");
        end
    endtask

    initial begin
        $dumpfile("pneumonia_8test.vcd");
        $dumpvars(0, tb_top);
    end

endmodule
