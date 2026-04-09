# ==============================================================================
# Timing Constraints for Pneumonia Detection Accelerator
# Board  : PYNQ-Z2  (xc7z020clg400-1)
# Target : 80 MHz (12.5 ns period)
#
# PYNQ-Z2 Notes:
#   - Device package is clg400 (not clg484 used on ZedBoard/PYNQ-Z1).
#   - The on-board 125 MHz oscillator drives pin H16 (MRCC capable).
#   - For a pure PL design at 80 MHz, route clk through a MMCM/PLL driven
#     from H16, OR bring 80 MHz in from an external source on a Pmod pin.
#   - For simulation / synthesis timing closure, the create_clock below is
#     sufficient. Assign [get_ports clk] to H16 (or your chosen input pin)
#     in the board-level XDC when moving to physical implementation.
#
# Pin assignments below follow the official PYNQ-Z2 master XDC.
# Uncomment the sections relevant to your physical I/O plan.
# ==============================================================================

# ------------------------------------------------------------------------------
# Primary clock constraint (applies to both synthesis and implementation)
# ------------------------------------------------------------------------------
create_clock -period 12.500 -name clk -waveform {0.000 6.250} [get_ports clk]

# ------------------------------------------------------------------------------
# Input / output delay constraints
# (keeps I/O timing analysis meaningful; adjust for your actual board paths)
# ------------------------------------------------------------------------------
set_input_delay  -clock [get_clocks clk] -min 0.500 [get_ports {rst start pixel_in[*] pixel_valid}]
set_input_delay  -clock [get_clocks clk] -max 2.000 [get_ports {rst start pixel_in[*] pixel_valid}]
set_output_delay -clock [get_clocks clk] -min -1.000 [get_ports {cancer_detected result_valid}]
set_output_delay -clock [get_clocks clk] -max  2.000 [get_ports {cancer_detected result_valid}]

# ------------------------------------------------------------------------------
# False path: async reset (driven from PS or button, not time-critical)
# ------------------------------------------------------------------------------
set_false_path -from [get_ports rst]

# ==============================================================================
# Physical pin assignments — PYNQ-Z2 (xc7z020clg400-1)
# ==============================================================================

# ------------------------------------------------------------------------------
# Clock - Using on-board 125 MHz oscillator
# NOTE: You'll need to add a Clock Wizard IP to divide 125 MHz down to 80 MHz
#       OR accept 125 MHz operation (change create_clock period to 8.0 ns above)
# ------------------------------------------------------------------------------
set_property PACKAGE_PIN H16          [get_ports clk]
set_property IOSTANDARD  LVCMOS33     [get_ports clk]

# ------------------------------------------------------------------------------
# Control inputs - Using on-board push-buttons
# BTN0 (D19) = rst, BTN1 (D20) = start
# ------------------------------------------------------------------------------
set_property PACKAGE_PIN D19          [get_ports rst]
set_property IOSTANDARD  LVCMOS33     [get_ports rst]
set_property PACKAGE_PIN D20          [get_ports start]
set_property IOSTANDARD  LVCMOS33     [get_ports start]

# ------------------------------------------------------------------------------
# Data input valid - Using Pmod JA pin 3
# ------------------------------------------------------------------------------
set_property PACKAGE_PIN Y16          [get_ports pixel_valid]
set_property IOSTANDARD  LVCMOS33     [get_ports pixel_valid]

# ------------------------------------------------------------------------------
# Pixel data input [7:0] - Using Pmod JB (all 8 pins)
# Top row: JB1-JB4 (bits 0-3), Bottom row: JB7-JB10 (bits 4-7)
# ------------------------------------------------------------------------------
set_property PACKAGE_PIN W14          [get_ports {pixel_in[0]}]
set_property PACKAGE_PIN Y14          [get_ports {pixel_in[1]}]
set_property PACKAGE_PIN T11          [get_ports {pixel_in[2]}]
set_property PACKAGE_PIN T10          [get_ports {pixel_in[3]}]
set_property PACKAGE_PIN V16          [get_ports {pixel_in[4]}]
set_property PACKAGE_PIN W16          [get_ports {pixel_in[5]}]
set_property PACKAGE_PIN V12          [get_ports {pixel_in[6]}]
set_property PACKAGE_PIN W13          [get_ports {pixel_in[7]}]
set_property IOSTANDARD  LVCMOS33     [get_ports {pixel_in[*]}]

# ------------------------------------------------------------------------------
# Output indicators - Using on-board LEDs
# LD0 (R14) = cancer_detected, LD1 (P14) = result_valid
# ------------------------------------------------------------------------------
set_property PACKAGE_PIN R14          [get_ports cancer_detected]
set_property IOSTANDARD  LVCMOS33     [get_ports cancer_detected]
set_property PACKAGE_PIN P14          [get_ports result_valid]
set_property IOSTANDARD  LVCMOS33     [get_ports result_valid]

# ==============================================================================
