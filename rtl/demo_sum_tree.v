// Auto-generated: 8-input registered adder tree, single-cycle.
module demo_sum_tree (
    input  wire clk,
    input  wire rst_n,
    input  wire [71:0] in0,
    input  wire [71:0] in1,
    input  wire [71:0] in2,
    input  wire [71:0] in3,
    input  wire [71:0] in4,
    input  wire [71:0] in5,
    input  wire [71:0] in6,
    input  wire [71:0] in7,
    output reg  [75:0] sum_o
);
    always @(posedge clk) begin
        if (!rst_n) sum_o <= 76'd0;
        else        sum_o <= in0 + in1 + in2 + in3 + in4 + in5 + in6 + in7;
    end
endmodule
