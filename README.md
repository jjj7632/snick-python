# snick_python

SoC side Python TCP and command scaffolding

## Files

- `../shared_protocol/soc_protocol.py`: command IDs, and master/slave mode logic
- `matlab_server_adapter.py`: TCP server on the SoC
- `../shared_protocol/numpysocket.py`: lightweight socket setup
- `fpga_buffer_manager.py`: PYNQ DDR buffer, AXI register, and DMA bridge scaffold

## Protocol

- `[1]`, `[10]`, `[30]`, `[98]`, `[99]`: command byte only
- `[11,n]`, `[12,n]`, `[15,frame]`: command byte + int32
- `[22,in_out]`: command byte + uint8
- `[21,frame,x,y,z]`: command byte + int32 + 3 float32 values
- `[50,frame,left_image,right_image]`: command byte + int32 + raw left image bytes + raw right image bytes

Default image format:

- `1920 x 1080 x 3`
- `uint8`

## Run

```bash
python matlab_server_adapter.py --host 0.0.0.0 --port 9999
```

With the FPGA fast path enabled, provide the overlay and DMA IP names:

```bash
sudo python3 matlab_server_adapter.py \
  --host 0.0.0.0 \
  --port 9999 \
  --image-width 1920 \
  --image-height 1080 \
  --image-channels 3 \
  --overlay-bit /home/xilinx/overlays/sparrow/sparrow.bit \
  --left-dma-name axi_dma_left \
  --right-dma-name axi_dma_right \
  --fpga-ip-base 0x43C60000
```

## Test

1. Start the server on the Snickerdoodle.
2. Set `BOARD_IP` in `matlab_client_demo.py`.
3. Run:

```bash
python matlab_client_demo.py
```

Note: Make sure to send the pixel arrays, not the raw `.png` file bytes.
