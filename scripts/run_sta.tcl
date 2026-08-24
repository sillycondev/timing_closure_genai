read_liberty /home/devansh/OpenSTA/test/sky130hd/sky130_fd_sc_hd__tt_025C_1v80.lib
read_verilog /home/devansh/OpenSTA/gcd_sky130hd.v
link_design gcd
read_sdc /home/devansh/OpenSTA/examples/gcd_sky130hd.sdc
report_checks -digits 4
exit
