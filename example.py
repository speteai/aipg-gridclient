"""Minimal gridclient example. Run:  AIPG_API_KEY=grid_... python3 example.py"""
import os
from gridclient import GridClient, GridError

g = GridClient(os.environ["AIPG_API_KEY"])

print("Text models :", g.models())
print("Image models:", g.image_models())

# Reasoning-safe chat (returns the answer, never an empty string on a reasoning model):
print("17*23 =", g.chat("qwen3:8b", "What is 17*23? Reply with just the number."))

# Generate an image (model is required):
try:
    imgs = g.generate_image("a single red circle on white", model="z-image-turbo", size="512x512")
    print("Image URL:", imgs[0].get("url"))
except GridError as e:
    print("Image gen skipped:", e)
