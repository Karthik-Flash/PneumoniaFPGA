`timescale 1ns / 1ps

module tb_top;

    // Clock and reset
    reg clk;
    reg rst;
    
    // DUT inputs
    reg start;
    reg [7:0] pixel_in;
    reg pixel_valid;
    
    // DUT outputs
    wire cancer_detected;
    wire result_valid;
    
    // Test control
    integer test_num;
    integer errors;
    integer pixel_idx;
    
    // Image memory (784 pixels = 28x28)
    reg [7:0] image_mem [0:783];

    // FSM state tracking (only print on CHANGE, not every cycle)
    reg [2:0] prev_fsm_state;
    
    // Clock generation: 100 MHz (10ns period)
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    // DUT instantiation
    top_accelerator dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .pixel_in(pixel_in),
        .pixel_valid(pixel_valid),
        .cancer_detected(cancer_detected),
        .result_valid(result_valid)
    );
    
    // Main test sequence
    initial begin
        $display("========================================");
        $display("TinyML Pneumonia Detection Accelerator");
        $display("BRAM-Optimized FPGA Testbench");
        $display("Target: Zynq-7020 (xc7z020clg484-1)");
        $display("========================================\n");
        
        // Initialize
        rst          = 1;
        start        = 0;
        pixel_in     = 8'h00;
        pixel_valid  = 0;
        errors       = 0;
        test_num     = 0;
        pixel_idx    = 0;
        prev_fsm_state = 3'bxxx;
        
        // Reset sequence
        repeat(10) @(posedge clk);
        rst = 0;
        repeat(5)  @(posedge clk);
        
        $display("[%0t ns] Reset released, beginning tests\n", $time);
        
        // Run test cases
        test_cancer_image("image_pneumonia1.mem",  1);
        test_cancer_image("image_pneumonia_v3.mem", 2);
        test_normal_image("image_normal1.mem",     1);
        test_normal_image("image_normal2.mem",     2);
        
        // Final summary
        #100;
        $display("\n========================================");
        $display("TEST SUMMARY");
        $display("========================================");
        $display("Tests Passed: %0d/%0d", test_num - errors, test_num);
        $display("Errors:       %0d", errors);
        
        if (errors == 0) begin
            $display("\n*** ALL TESTS PASSED - BRAM design verified ***");
        end else begin
            $display("\n*** %0d TEST(S) FAILED ***", errors);
        end
        
        $display("========================================\n");
        $finish;
    end
    
    // -------------------------------------------------------
    // FSM state monitor: prints ONLY on state CHANGE
    // (Fixes the millions-of-lines freeze in original code)
    // -------------------------------------------------------
    always @(posedge clk) begin
        if (dut.u_fsm.state !== prev_fsm_state) begin
            case (dut.u_fsm.state)
                3'b000: $display("[%0t ns] FSM -> IDLE",   $time);
                3'b001: $display("[%0t ns] FSM -> CONV",   $time);
                3'b010: $display("[%0t ns] FSM -> POOL",   $time);
                3'b011: $display("[%0t ns] FSM -> FC1",    $time);
                3'b100: $display("[%0t ns] FSM -> FC2",    $time);
                3'b101: $display("[%0t ns] FSM -> OUTPUT", $time);
                default: $display("[%0t ns] FSM -> UNKNOWN (%b)", $time, dut.u_fsm.state);
            endcase
            prev_fsm_state <= dut.u_fsm.state;
        end
    end

    // -------------------------------------------------------
    // Global watchdog — catches any remaining infinite loops
    // 4 inferences x 500k cycles max = 2M cycles = 20ms
    // -------------------------------------------------------
    initial begin
        #200000000; // 200ms absolute wall limit (200,000,000 ns)
        $display("\n*** GLOBAL WATCHDOG TIMEOUT after 200ms ***");
        $finish;
    end
    
    // -------------------------------------------------------
    // Task: Pneumonia-positive images
    // -------------------------------------------------------
    task test_cancer_image;
        input [1000*8-1:0] filename;
        input integer img_num;
        integer timeout_counter;
        begin
            test_num = test_num + 1;
            repeat(50) @(posedge clk); // brief inter-test gap
            
            $display("========================================");
            $display("[Test %0d] Pneumonia Image %0d", test_num, img_num);
            $readmemh(filename, image_mem);
            $display("  Loaded: %0s", filename);
            
            // Pulse start
            @(posedge clk); start = 1;
            @(posedge clk); start = 0;
            
            // Stream 784 pixels
            for (pixel_idx = 0; pixel_idx < 784; pixel_idx = pixel_idx + 1) begin
                @(posedge clk);
                pixel_in    = image_mem[pixel_idx];
                pixel_valid = 1;
            end
            @(posedge clk);
            pixel_valid = 0;
            pixel_in    = 8'h00;
            $display("  Streamed 784 pixels. Waiting for result...");
            
            // Wait with per-test timeout (Verilog-2001 compatible)
            timeout_counter = 0;
            while (result_valid !== 1'b1 && timeout_counter < 2000000) begin
                @(posedge clk);
                timeout_counter = timeout_counter + 1;
            end
            
            if (timeout_counter >= 500000) begin
                $display("*** TIMEOUT on Test %0d - result_valid never high ***", test_num);
                $finish;
            end
            
            repeat(2) @(posedge clk);
            
            if (cancer_detected === 1'b1) begin
                $display("  [PASS] cancer_detected=1 (pneumonia correct)");
            end else begin
                $display("  [FAIL] cancer_detected=0 (missed pneumonia!)");
                errors = errors + 1;
            end
            $display("========================================\n");
        end
    endtask
    
    // -------------------------------------------------------
    // Task: Normal (pneumonia-negative) images
    // -------------------------------------------------------
    task test_normal_image;
        input [1000*8-1:0] filename;
        input integer img_num;
        integer timeout_counter;
        begin
            test_num = test_num + 1;
            repeat(50) @(posedge clk);
            
            $display("========================================");
            $display("[Test %0d] Normal Image %0d", test_num, img_num);
            $readmemh(filename, image_mem);
            $display("  Loaded: %0s", filename);
            
            // Pulse start
            @(posedge clk); start = 1;
            @(posedge clk); start = 0;
            
            // Stream 784 pixels
            for (pixel_idx = 0; pixel_idx < 784; pixel_idx = pixel_idx + 1) begin
                @(posedge clk);
                pixel_in    = image_mem[pixel_idx];
                pixel_valid = 1;
            end
            @(posedge clk);
            pixel_valid = 0;
            pixel_in    = 8'h00;
            $display("  Streamed 784 pixels. Waiting for result...");
            
            // Wait with per-test timeout (Verilog-2001 compatible)
            timeout_counter = 0;
            while (result_valid !== 1'b1 && timeout_counter < 2000000) begin
                @(posedge clk);
                timeout_counter = timeout_counter + 1;
            end
            
            if (timeout_counter >= 500000) begin
                $display("*** TIMEOUT on Test %0d - result_valid never high ***", test_num);
                $finish;
            end
            
            repeat(2) @(posedge clk);
            
            if (cancer_detected === 1'b0) begin
                $display("  [PASS] cancer_detected=0 (normal correct)");
            end else begin
                $display("  [FAIL] cancer_detected=1 (false positive!)");
                errors = errors + 1;
            end
            $display("========================================\n");
        end
    endtask
    
    // Waveform dump
    initial begin
        $dumpfile("pneumonia_bram.vcd");
        $dumpvars(0, tb_top);
    end

endmodule
