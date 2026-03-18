import os
import sys
import git

if "APP_PATH" in os.environ:
    # fix sys.path for import
    os.chdir(os.environ["APP_PATH"])
    if os.getcwd() not in sys.path:
        sys.path.append(os.getcwd())

# remove duplicate gradio_app path from sys.path
sys.path = list(dict.fromkeys(sys.path))

# remove gradio reload env if in huggingface space
if "SPACE_ID" in os.environ:
    for key in ["GRADIO_WATCH_DIRS", "GRADIO_WATCH_MODULE_NAME", "GRADIO_WATCH_DEMO_NAME", "GRADIO_WATCH_DEMO_PATH"]:
        if key in os.environ:
            del os.environ[key]

def get_app_git_commit():
    app_path = os.environ.get("APP_PATH")
    if not app_path:
        return None
    try:
        repo = git.Repo(app_path, search_parent_directories=False)
        hexsha = repo.head.commit.hexsha
        return hexsha
    except (git.exc.InvalidGitRepositoryError, ValueError, git.exc.GitError):
        return None

# here the subprocess stops loading, because __name__ is NOT '__main__'
# gradio will reload
if '__main__' == __name__:

    import gradio as gr
    from contextlib import suppress

    import os
    import torch
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF
    import torch.nn.functional as F
    import json
    from PIL import Image

    from dall_e          import map_pixels, unmap_pixels, load_model

    CHECKPOINT_MODEL_PATH = os.environ.get("CHECKPOINT_MODEL_PATH", './')

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = load_model(f"{CHECKPOINT_MODEL_PATH}/encoder.pkl", device)
    dec = load_model(f"{CHECKPOINT_MODEL_PATH}/decoder.pkl", device)

    target_image_size = 256
    transform = T.Compose([
        T.Resize(target_image_size, interpolation=Image.LANCZOS),
        T.CenterCrop(target_image_size),
        T.ToTensor(),
    ])

    def encode_image(image):
        """
        Encodes the given image.

        Args:
            image (Union[PIL.Image.Image, str]): Input image, can be a PIL Image object, image array, or URL.

        Returns:
            dict: A dictionary with the following fields:
                - ``shape`` (List[int]): Spatial size of the code grid, fixed as ``[32, 32]``.
                - ``vocab_size`` (int): Vocabulary size of the model, fixed as `8192`.
                - ``tokens`` (List[List[int]]): Quantized image tokens on a ``32 x 32`` grid,
                  converted from the internal one-hot representation to nested Python lists
                  on CPU for serialization and front-end display.
        """
        if image is None:
            raise gr.Error("Please upload an image before clicking the \"Encoding\" button.")

        if isinstance(image, Image.Image) is False:
            image = Image.fromarray(image)

        x = transform(image).to(device)
        x = map_pixels(x.unsqueeze(0))

        z_logits = enc(x)
        z = torch.argmax(z_logits, axis=1)

        z_cpu = z.squeeze(0).cpu().numpy().tolist()
        return {
            "shape": [32, 32],
            "vocab_size": enc.vocab_size,
            "tokens": z_cpu
        }

    def decode_code(code):
        """
        Embeds one or more watermarks into the input image.

        Args:
            code (Union[str, dict]): Encoded image code. It can be a JSON string or a
                Python dictionary with the same structure as the output of ``encode_image``,
                i.e. containing at least a ``"tokens"`` field that holds the quantized
                image tokens.

        Returns:
            image (Union[PIL.Image.Image, str]): Output image, either a PIL Image object or a URL pointing to the image.
        """
        if isinstance(code, str):
            code = json.loads(code)

        # code["tokens"] is expected to be a 32x32 grid of token ids
        # match the shape used in process_image: [1, 32, 32] -> one-hot -> [1, vocab, 32, 32]
        z = torch.tensor(code["tokens"], dtype=torch.long).unsqueeze(0).to(device)  # [1,32,32]
        z = F.one_hot(z, num_classes=enc.vocab_size).permute(0, 3, 1, 2).float()  # [1,vocab,32,32]

        x_stats = dec(z).float()
        x_rec = unmap_pixels(torch.sigmoid(x_stats[:, :3]))
        x_rec = T.ToPILImage(mode='RGB')(x_rec[0].cpu())

        return x_rec

    def process_image(image):
        if isinstance(image, Image.Image) is False:
            image = Image.fromarray(image)

        x = transform(image).to(device)
        x = map_pixels(x.unsqueeze(0))

        z_logits = enc(x)
        z = torch.argmax(z_logits, axis=1)


        z = F.one_hot(z, num_classes=enc.vocab_size).permute(0, 3, 1, 2).float()

        x_stats = dec(z).float()
        x_rec = unmap_pixels(torch.sigmoid(x_stats[:, :3]))
        x_rec = T.ToPILImage(mode='RGB')(x_rec[0].to('cpu'))

        return x_rec

    DALLE_VERSION = get_app_git_commit() or "unknown"

    with gr.Blocks(title="DALL- Demo", css="""
            .align-bottom {display:flex; flex-direction:column; justify-content:flex-end;}
            """) as demo:
        gr.Markdown(f"""
        # DALL-E Demo

        > DALL-E: [`{DALLE_VERSION}`](https://github.com/openai/DALL-E/tree/{DALLE_VERSION})

        Find the original project [here](https://github.com/openai/DALL-E).
        Or this project [here](https://github.com/xiaoyao9184/docker-dall-e).
        See the [README](https://huggingface.co/spaces/xiaoyao9184/dall-e/blob/main/README.md) for Spaces's metadata.
        """)

        with gr.Tabs():
            with gr.TabItem("Encoding -> Decoding"):
                with gr.Row():
                    with gr.Column():
                        original_img = gr.Image(label="Original Image", type="numpy", height=512)
                        process_btn = gr.Button("process")
                    with gr.Column():
                        reconstructed_image = gr.Image(label="Reconstructed Image")
            with gr.TabItem("Encoding"):
                with gr.Row():
                    with gr.Column():
                        encoding_img = gr.Image(label="Input Image", type="numpy", height=512)
                        encoding_btn = gr.Button("Encoding")
                    with gr.Column():
                        encoding_messages = gr.JSON(label="Encoding Messages")
            with gr.TabItem("Decoding"):
                with gr.Row():
                    with gr.Column():
                        encoding_code = gr.Code(label="Encoding Code", language="json")
                        decoding_btn = gr.Button("Decoding")
                    with gr.Column(elem_classes="align-bottom"):
                        decoding_image = gr.Image(label="Decoding Image")

                encoding_btn.click(
                    fn=encode_image,
                    inputs=encoding_img,
                    outputs=encoding_messages
                )
                decoding_btn.click(
                    fn=decode_code,
                    inputs=encoding_code,
                    outputs=decoding_image
                )
                process_btn.click(
                    fn=process_image,
                    inputs=[original_img],
                    outputs=[reconstructed_image],
                    api_name=False
                )

    if __name__ == '__main__':
        demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
