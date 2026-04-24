// Top-level wrapper for the Pipelined QNN Accelerator
// Provides standardized Handshaking (valid_in/valid_out) and I/O bundling.

module top (
    input  wire        clk,
    input  wire        rst,
    input  wire        valid_in,
    input  wire [31:0] features,
    output wire        valid_out,
    output wire [23:0] prediction
);

    wire [7:0] score0, score1, score2;
    reg [3:0]  v_delay;

    // Instantiate the pipelined accelerator
    toplevel u_accel (
        .clk(clk),
        .rst(rst),
        .features(features),
        .score0(score0),
        .score1(score1),
        .score2(score2)
    );

    // Handshaking Logic: Delay valid_in by the pipeline depth (4 cycles)
    // Stage 0: Input Reg, Stage 1: L1, Stage 2: L2, Stage 3: Out Reg
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            v_delay <= 4'b0;
        end else begin
            v_delay <= {v_delay[2:0], valid_in};
        end
    end

    assign valid_out = v_delay[3];

    // Bundle outputs into Big-Endian prediction bus
    // [Score0(23:16), Score1(15:8), Score2(7:0)]
    assign prediction = {score0, score1, score2};

endmodule
