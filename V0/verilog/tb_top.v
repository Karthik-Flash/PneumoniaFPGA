`timescale 1ns / 1ps
module tb_top;

    reg clk;
    reg rst;
    reg start;
    reg [7:0] pixel_in;
    reg pixel_valid;
    wire cancer_detected;
    wire result_valid;
    
    integer test_num;
    integer errors;
    integer pixel_idx;
    
    reg [7:0] image_mem [0:783];
    
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    top_accelerator dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .pixel_in(pixel_in),
        .pixel_valid(pixel_valid),
        .cancer_detected(cancer_detected),
        .result_valid(result_valid)
    );
    
    initial begin
        $display("========================================");
        $display("TinyML Pneumonia Detection Accelerator");
        $display("FPGA Testbench");
        $display("========================================\n");
        
        rst = 1;
        start = 0;
        pixel_in = 8'h00;
        pixel_valid = 0;
        errors = 0;
        test_num = 0;
        pixel_idx = 0;
        
        repeat(5) @(posedge clk);
        rst = 0;
        repeat(5) @(posedge clk);
        
        $display("[%0t ns] Reset released, beginning tests\n", $time);
        
        test_cancer_image("image_pneumonia1.mem", 1);
        test_cancer_image("image_pneumonia_v2.mem", 2);
        test_normal_image("image_normal1.mem", 1);
        test_normal_image("image_normal2.mem", 2);
        
        #100;
        $display("\n========================================");
        $display("TEST SUMMARY");
        $display("========================================");
        $display("Tests Passed: %0d/4", test_num - errors);
        $display("Total Tests:  %0d", test_num);
        $display("Errors:       %0d", errors);
        
        if (errors == 0) begin
            $display("\n*** ALL TESTS PASSED ***");
            $display("Pneumonia detection working correctly!");
        end else begin
            $display("\n*** TESTS FAILED ***");
            $display("Found %0d error(s)", errors);
        end
        $display("========================================\n");
        $finish;
    end
    
    task test_cancer_image;
        input [1000*8-1:0] filename;
        input integer img_num;
        begin
            test_num = test_num + 1;

            // Wait for FSM to be IDLE before starting new test
            repeat(100) @(posedge clk);

            $display("========================================");
            $display("[Test %0d] Pneumonia Image %0d", test_num, img_num);
            $display("========================================");
            $display("Loading: %0s", filename);
            
            $readmemh(filename, image_mem);
            $display("  [OK] Loaded 784 pixels");
            
            // Pulse start
            @(posedge clk);
            start = 1;
            @(posedge clk);
            start = 0;
            
            $display("  Streaming pixels...");
            for (pixel_idx = 0; pixel_idx < 784; pixel_idx = pixel_idx + 1) begin
                @(posedge clk);
                pixel_in = image_mem[pixel_idx];
                pixel_valid = 1;
            end
            
            @(posedge clk);
            pixel_valid = 0;
            pixel_in = 8'h00;
            $display("  [OK] All 784 pixels streamed");
            
            $display("  Waiting for inference to complete...");
            wait(result_valid == 1);
            repeat(2) @(posedge clk);
            
            $display("\n--- RESULTS ---");
            $display("cancer_detected: %b", cancer_detected);
            $display("Expected:        1 (pneumonia)");
            
            if (cancer_detected === 1'b1) begin
                $display("  [PASS] Correctly detected pneumonia");
            end else begin
                $display("  [FAIL] Should have detected pneumonia!");
                errors = errors + 1;
            end
            $display("========================================\n");
            
            repeat(20) @(posedge clk);
        end
    endtask
    
    task test_normal_image;
        input [1000*8-1:0] filename;
        input integer img_num;
        begin
            test_num = test_num + 1;

            // Wait for FSM to be IDLE before starting new test
            repeat(100) @(posedge clk);

            $display("========================================");
            $display("[Test %0d] Normal Image %0d", test_num, img_num);
            $display("========================================");
            $display("Loading: %0s", filename);
            
            $readmemh(filename, image_mem);
            $display("  [OK] Loaded 784 pixels");
            
            // Pulse start
            @(posedge clk);
            start = 1;
            @(posedge clk);
            start = 0;
            
            $display("  Streaming pixels...");
            for (pixel_idx = 0; pixel_idx < 784; pixel_idx = pixel_idx + 1) begin
                @(posedge clk);
                pixel_in = image_mem[pixel_idx];
                pixel_valid = 1;
            end
            
            @(posedge clk);
            pixel_valid = 0;
            pixel_in = 8'h00;
            $display("  [OK] All 784 pixels streamed");
            
            $display("  Waiting for inference to complete...");
            wait(result_valid == 1);
            repeat(2) @(posedge clk);
            
            $display("\n--- RESULTS ---");
            $display("cancer_detected: %b", cancer_detected);
            $display("Expected:        0 (normal/benign)");
            
            if (cancer_detected === 1'b0) begin
                $display("  [PASS] Correctly classified as normal");
            end else begin
                $display("  [FAIL] False positive - flagged normal as pneumonia!");
                errors = errors + 1;
            end
            $display("========================================\n");
            
            repeat(20) @(posedge clk);
        end
    endtask
    
    initial begin
        $dumpfile("pneumonia_detection.vcd");
        $dumpvars(0, tb_top);
    end
    
    // Timeout: 600ms covers 4 full inferences
    initial begin
        #600000000;
        $display("\n*** ERROR: Simulation timeout! ***");
        $finish;
    end
    
    // FSM monitor
    always @(posedge clk) begin
        if (dut.fsm_state == 3'b001)
            $display("[%0t ns] FSM: CONV state", $time);
        else if (dut.fsm_state == 3'b010)
            $display("[%0t ns] FSM: POOL state", $time);
        else if (dut.fsm_state == 3'b011)
            $display("[%0t ns] FSM: FC1 state", $time);
        else if (dut.fsm_state == 3'b100)
            $display("[%0t ns] FSM: FC2 state", $time);
        else if (dut.fsm_state == 3'b101)
            $display("[%0t ns] FSM: OUTPUT state", $time);
    end

endmodule