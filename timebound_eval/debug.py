"""Script to check general video language test."""
import torch
from languagebind import LanguageBindVideo, LanguageBindVideoTokenizer, LanguageBindVideoProcessor

pretrained_ckpt = 'LanguageBind/LanguageBind_Video_FT'  # also 'LanguageBind/LanguageBind_Video'
model = LanguageBindVideo.from_pretrained(pretrained_ckpt)
tokenizer = LanguageBindVideoTokenizer.from_pretrained(pretrained_ckpt)
video_process = LanguageBindVideoProcessor(model.config, tokenizer)

model.eval()
data = video_process(
    ["timebound_eval/folding_paper.mp4"],
    ['Someone crushing a paper.'],
    return_tensors='pt',
)
with torch.no_grad():
    out = model(**data)

print(out.text_embeds @ out.image_embeds.T)