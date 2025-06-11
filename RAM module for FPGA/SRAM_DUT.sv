`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 01/11/2025 05:26:02 PM
// Design Name: 
// Module Name: SRAM_DUT
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: SRAM Testbench with fixed issues
// 
//////////////////////////////////////////////////////////////////////////////////

`timescale 1ns / 1ps

module SRAM_tb;

    // Inputs to SRAM
    reg clk;
    reg reset;
    reg write_enable;
    reg [7:0] data_in;
    reg [7:0] address;

    // Output from SRAM
    wire [7:0] data_out;

    // Instantiate the SRAM module
    SRAM uut (
        .clk(clk),
        .reset(reset),
        .write_enable(write_enable),
        .data_in(data_in),
        .address(address),
        .data_out(data_out)
    );

    // Clock generation
    always #5 clk = ~clk;  // 10ns period

    initial begin
        // Initialize signals
        clk = 0;
        reset = 0;
        write_enable = 0;
        data_in = 8'b0;
        address = 8'b0;

        // Reset the SRAM
        #10 reset = 1;  // Assert reset for 10ns
        #10 reset = 0;   // Deassert reset

        // Test write operation

        data_in = 8'h55;  // Write value 0x55
        address = 8'h10;  // Write to address 0x10
        write_enable = 1;
        #10
        write_enable = 0; // Disable write -> enable read

        // Test read operation
        address = 8'h10;  // Read from address 0x10
        #10;

        // Test another write
        
        data_in = 8'hA5;  // Write value 0xA5
        address = 8'h20;  // Write to address 0x20
        write_enable = 1;  
        #10; 
        write_enable = 0; // Disable write

        // Test read after second write
        address = 8'h20;  // Read from address 0x20
        #10;

        // Test read from an address with no data written
        address = 8'h30;  // Read from an empty address initialized 0 by reset
        
        // Reset the SRAM
        #10; reset = 1;  // Assert reset for 10ns
        #10; reset = 0;   // Deassert reset
        
        address = 8'h20; // will read 0 after reset, write anabled is  0 so data in will not  be written.
        #10;

        $finish;
    end

    // Monitor data_out for debugging
    initial begin
        $monitor("At time %t: data_out = %h", $time, data_out);
    end

endmodule
