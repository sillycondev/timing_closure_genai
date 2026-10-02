// Demo design: 8-input 32-bit accumulator written as a serial adder chain.
// The whole chain sits between one register stage, so the critical path
// runs through 7 back-to-back adders and fails a 1 GHz clock badly.
module demo_adder_chain (
    input  wire clk,
    input  wire rst_n,
    input  wire [31:0] in0,
    input  wire [31:0] in1,
    input  wire [31:0] in2,
    input  wire [31:0] in3,
    input  wire [31:0] in4,
    input  wire [31:0] in5,
    input  wire [31:0] in6,
    input  wire [31:0] in7,
    output reg  [34:0] sum_o
);
    always @(posedge clk) begin
        if (!rst_n) sum_o <= 35'd0;
        else        sum_o <= in0 + in1 + in2 + in3 + in4 + in5 + in6 + in7;
    end
endmodule
