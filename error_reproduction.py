from peft import LoraConfig, get_peft_model
from visual import VisionTransformer

if __name__ == '__main__':
    model = VisionTransformer()

    target_modules = ['attn_pool.attn', 'attn_pool.kv_proj']
    peft_config = LoraConfig(
        target_modules=target_modules,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05
    )
    model = get_peft_model(model, peft_config)

    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    model.base_model.attn_pool.attn.base_layer.in_proj_weight.requires_grad = False
    model.base_model.attn_pool.attn.base_layer.out_proj.base_layer.weight.requires_grad = False

    trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]

    print(trainable_params)
    print(model.base_model.attn_pool.attn.base_layer.in_proj_weight.requires_grad)
