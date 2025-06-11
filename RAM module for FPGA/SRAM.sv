`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 01/11/2025 04:42:58 PM
// Design Name: 
// Module Name: SRAM
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module SRAM(

input clk, reset, write_enable,
input [7:0] data_in, address,
output reg [7:0] data_out
);

reg [7:0] memory [256];
integer i;
always @(negedge clk) begin
    
    if(reset) begin
        for(i = 0; i <256; i++) begin
            memory[i] = 0;
        end
    end
    else if (write_enable) begin
        memory[address] = data_in;
        end

    else begin

        assign data_out = memory[address];

    end

end

endmodule



