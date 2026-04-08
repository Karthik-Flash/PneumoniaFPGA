`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Module Name: tb_bram_verify
// Description: Testbench to verify BRAM read-after-write correctness
//              Tests 1-cycle latency handling in feature_maps and pooled_maps
//
// Test Coverage:
//   1. Write-then-read: Verify data integrity with 1-cycle latency
//   2. Multi-channel addressing: Test all 4 channels independently
//   3. Sequential access: Verify pooling read patterns
//   4. Boundary conditions: Test first/last addresses
//
// Expected Results:
//   - All written data should be readable after 1-cycle latency
//   - No data corruption across channel boundaries
//   - Correct addressing for multi-dimensional indexing
//
//////////////////////////////////////////////////////////////////////////////////

module tb_bram_verify;

    // Clock and reset
    reg clk;
    reg rst;
    
    // Feature maps BRAM signals (3,136 × 20-bit)
    (* ram_style = "block" *) reg [19:0] feature_maps_bram [0:3135];
    reg [11:0] fm_wr_addr;
    reg [19:0] fm_wr_data;
    reg fm_wr_en;
    reg [11:0] fm_rd_addr;
    reg [19:0] fm_rd_data;
    
    // Pooled maps BRAM signals (784 × 20-bit)
    (* ram_style = "block" *) reg [19:0] pooled_maps_bram [0:783];
    reg [9:0] pm_wr_addr;
    reg [19:0] pm_wr_data;
    reg pm_wr_en;
    reg [9:0] pm_rd_addr;
    reg [19:0] pm_rd_data;
    
    // Test control
    integer test_num;
    integer errors;
    integer i, j, k, ch;
    
    // BRAM models (same as in top_accelerator)
    always @(posedge clk) begin
        if (fm_wr_en) begin
            feature_maps_bram[fm_wr_addr] <= fm_wr_data;
        end
    end
    
    always @(posedge clk) begin
        fm_rd_data <= feature_maps_bram[fm_rd_addr];
    end
    
    always @(posedge clk) begin
        if (pm_wr_en) begin
            pooled_maps_bram[pm_wr_addr] <= pm_wr_data;
        end
    end
    
    always @(posedge clk) begin
        pm_rd_data <= pooled_maps_bram[pm_rd_addr];
    end
    
    // Clock generation: 100 MHz (10ns period)
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    // Test sequence
    initial begin
        $display("========================================");
        $display("  BRAM Read-After-Write Verification");
        $display("========================================");
        
        // Initialize
        rst = 1;
        fm_wr_en = 0;
        fm_wr_addr = 0;
        fm_wr_data = 0;
        fm_rd_addr = 0;
        pm_wr_en = 0;
        pm_wr_addr = 0;
        pm_wr_data = 0;
        pm_rd_addr = 0;
        test_num = 0;
        errors = 0;
        
        #20;
        rst = 0;
        #10;
        
        //----------------------------------------------------------------------
        // TEST 1: Basic Write-Then-Read (Feature Maps)
        //----------------------------------------------------------------------
        test_num = 1;
        $display("\n[TEST %0d] Basic write-then-read test (feature_maps_bram)", test_num);
        
        // Write test pattern to channel 0, position (0,0)
        @(posedge clk);
        fm_wr_en = 1;
        fm_wr_addr = 12'd0;  // Channel 0, row 0, col 0
        fm_wr_data = 20'h12345;
        
        @(posedge clk);
        fm_wr_en = 0;
        fm_rd_addr = 12'd0;  // Issue read
        
        @(posedge clk);
        @(posedge clk);  // Wait for 1-cycle latency
        
        if (fm_rd_data == 20'h12345) begin
            $display("  PASS: Read data matches written data (0x%05h)", fm_rd_data);
        end else begin
            $display("  FAIL: Expected 0x12345, got 0x%05h", fm_rd_data);
            errors = errors + 1;
        end
        
        //----------------------------------------------------------------------
        // TEST 2: Multi-Channel Addressing (Feature Maps)
        //----------------------------------------------------------------------
        test_num = 2;
        $display("\n[TEST %0d] Multi-channel addressing test (4 channels)", test_num);
        
        // Write unique pattern to each channel at position (5,7)
        for (ch = 0; ch < 4; ch = ch + 1) begin
            @(posedge clk);
            fm_wr_en = 1;
            fm_wr_addr = (ch * 784) + (5 * 28) + 7;  // Channel offset + row*28 + col
            fm_wr_data = 20'hA0000 | (ch << 12) | ((5 << 6) | 7);  // Encoded ch:row:col
        end
        
        @(posedge clk);
        fm_wr_en = 0;
        
        // Read back and verify
        for (ch = 0; ch < 4; ch = ch + 1) begin
            @(posedge clk);
            fm_rd_addr = (ch * 784) + (5 * 28) + 7;
            
            @(posedge clk);
            @(posedge clk);  // Wait for latency
            
            if (fm_rd_data == (20'hA0000 | (ch << 12) | ((5 << 6) | 7))) begin
                $display("  PASS: Channel %0d data correct (0x%05h)", ch, fm_rd_data);
            end else begin
                $display("  FAIL: Channel %0d expected 0x%05h, got 0x%05h", 
                         ch, (20'hA0000 | (ch << 12) | ((5 << 6) | 7)), fm_rd_data);
                errors = errors + 1;
            end
        end
        
        //----------------------------------------------------------------------
        // TEST 3: Sequential Write Burst (Convolution Pattern)
        //----------------------------------------------------------------------
        test_num = 3;
        $display("\n[TEST %0d] Sequential write burst (simulates convolution)", test_num);
        
        // Write first 10 locations of channel 2
        for (i = 0; i < 10; i = i + 1) begin
            @(posedge clk);
            fm_wr_en = 1;
            fm_wr_addr = (2 * 784) + i;  // Channel 2, linear address
            fm_wr_data = 20'h30000 | i;
        end
        
        @(posedge clk);
        fm_wr_en = 0;
        
        // Read back with burst
        for (i = 0; i < 10; i = i + 1) begin
            @(posedge clk);
            fm_rd_addr = (2 * 784) + i;
            
            @(posedge clk);
            @(posedge clk);  // Wait for latency
            
            if (fm_rd_data == (20'h30000 | i)) begin
                $display("  PASS: Address %0d data correct (0x%05h)", i, fm_rd_data);
            end else begin
                $display("  FAIL: Address %0d expected 0x%05h, got 0x%05h", 
                         i, (20'h30000 | i), fm_rd_data);
                errors = errors + 1;
            end
        end
        
        //----------------------------------------------------------------------
        // TEST 4: Pooled Maps Basic Read-After-Write
        //----------------------------------------------------------------------
        test_num = 4;
        $display("\n[TEST %0d] Pooled maps write-then-read test", test_num);
        
        // Write test pattern to all 4 channels at position (3,4)
        for (ch = 0; ch < 4; ch = ch + 1) begin
            @(posedge clk);
            pm_wr_en = 1;
            pm_wr_addr = (ch * 196) + (3 * 14) + 4;  // Channel offset + row*14 + col
            pm_wr_data = 20'hB0000 | (ch << 12) | ((3 << 6) | 4);
        end
        
        @(posedge clk);
        pm_wr_en = 0;
        
        // Read back and verify
        for (ch = 0; ch < 4; ch = ch + 1) begin
            @(posedge clk);
            pm_rd_addr = (ch * 196) + (3 * 14) + 4;
            
            @(posedge clk);
            @(posedge clk);  // Wait for latency
            
            if (pm_rd_data == (20'hB0000 | (ch << 12) | ((3 << 6) | 4))) begin
                $display("  PASS: Pooled channel %0d data correct (0x%05h)", ch, pm_rd_data);
            end else begin
                $display("  FAIL: Pooled channel %0d expected 0x%05h, got 0x%05h", 
                         ch, (20'hB0000 | (ch << 12) | ((3 << 6) | 4)), pm_rd_data);
                errors = errors + 1;
            end
        end
        
        //----------------------------------------------------------------------
        // TEST 5: 2x2 Window Read Pattern (Pooling Simulation)
        //----------------------------------------------------------------------
        test_num = 5;
        $display("\n[TEST %0d] 2×2 window read pattern (simulates pooling)", test_num);
        
        // Write 2x2 window to feature map channel 1 at position (4,6)
        // This simulates a pooling window that will be max-pooled
        for (i = 0; i < 2; i = i + 1) begin  // Row offset
            for (j = 0; j < 2; j = j + 1) begin  // Col offset
                @(posedge clk);
                fm_wr_en = 1;
                fm_wr_addr = (1 * 784) + ((4 + i) * 28) + (6 + j);
                fm_wr_data = 20'h40000 | (i << 8) | j;  // Encode position in data
            end
        end
        
        @(posedge clk);
        fm_wr_en = 0;
        
        // Read back 2x2 window (simulates pooling read)
        $display("  Reading 2×2 window from channel 1, pool position (2,3):");
        for (i = 0; i < 2; i = i + 1) begin
            for (j = 0; j < 2; j = j + 1) begin
                @(posedge clk);
                fm_rd_addr = (1 * 784) + ((4 + i) * 28) + (6 + j);
                
                @(posedge clk);
                @(posedge clk);  // Wait for latency
                
                if (fm_rd_data == (20'h40000 | (i << 8) | j)) begin
                    $display("    PASS: Position (%0d,%0d) = 0x%05h", i, j, fm_rd_data);
                end else begin
                    $display("    FAIL: Position (%0d,%0d) expected 0x%05h, got 0x%05h", 
                             i, j, (20'h40000 | (i << 8) | j), fm_rd_data);
                    errors = errors + 1;
                end
            end
        end
        
        //----------------------------------------------------------------------
        // TEST 6: Boundary Conditions
        //----------------------------------------------------------------------
        test_num = 6;
        $display("\n[TEST %0d] Boundary condition tests", test_num);
        
        // Test first address (0)
        @(posedge clk);
        fm_wr_en = 1;
        fm_wr_addr = 12'd0;
        fm_wr_data = 20'hAAAAA;
        
        // Test last address (3135)
        @(posedge clk);
        fm_wr_addr = 12'd3135;
        fm_wr_data = 20'h55555;
        
        @(posedge clk);
        fm_wr_en = 0;
        
        // Read first address
        @(posedge clk);
        fm_rd_addr = 12'd0;
        
        @(posedge clk);
        @(posedge clk);
        
        if (fm_rd_data == 20'hAAAAA) begin
            $display("  PASS: First address (0) = 0x%05h", fm_rd_data);
        end else begin
            $display("  FAIL: First address expected 0xAAAAA, got 0x%05h", fm_rd_data);
            errors = errors + 1;
        end
        
        // Read last address
        @(posedge clk);
        fm_rd_addr = 12'd3135;
        
        @(posedge clk);
        @(posedge clk);
        
        if (fm_rd_data == 20'h55555) begin
            $display("  PASS: Last address (3135) = 0x%05h", fm_rd_data);
        end else begin
            $display("  FAIL: Last address expected 0x55555, got 0x%05h", fm_rd_data);
            errors = errors + 1;
        end
        
        //----------------------------------------------------------------------
        // TEST 7: Rapid Write-Read Cycles
        //----------------------------------------------------------------------
        test_num = 7;
        $display("\n[TEST %0d] Rapid write-read cycles (back-to-back access)", test_num);
        
        for (i = 0; i < 5; i = i + 1) begin
            // Write
            @(posedge clk);
            fm_wr_en = 1;
            fm_wr_addr = 100 + i;
            fm_wr_data = 20'h60000 | i;
            
            @(posedge clk);
            fm_wr_en = 0;
            
            // Immediate read
            fm_rd_addr = 100 + i;
            
            @(posedge clk);
            @(posedge clk);  // Wait for latency
            
            if (fm_rd_data == (20'h60000 | i)) begin
                $display("  PASS: Rapid cycle %0d correct", i);
            end else begin
                $display("  FAIL: Rapid cycle %0d expected 0x%05h, got 0x%05h", 
                         i, (20'h60000 | i), fm_rd_data);
                errors = errors + 1;
            end
        end
        
        //----------------------------------------------------------------------
        // Final Results
        //----------------------------------------------------------------------
        #100;
        
        $display("\n========================================");
        $display("  TEST SUMMARY");
        $display("========================================");
        $display("  Total tests run: %0d", test_num);
        $display("  Total errors:    %0d", errors);
        
        if (errors == 0) begin
            $display("\n  *** ALL TESTS PASSED ***");
            $display("  BRAM read-after-write verification successful!");
            $display("  1-cycle latency correctly handled.");
        end else begin
            $display("\n  *** %0d TEST(S) FAILED ***", errors);
            $display("  Please review BRAM implementation.");
        end
        
        $display("========================================\n");
        
        $finish;
    end
    
    // Timeout watchdog
    initial begin
        #100000;  // 100us timeout
        $display("\nERROR: Testbench timeout!");
        $finish;
    end

endmodule
