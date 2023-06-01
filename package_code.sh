python -m zipapp toadgan -m "celeste.onnx_read:main" -o celestegan.pyz
zip -ur celestegan.pyz data/mask_small.json.gz data/generators data/upscale.onnx
