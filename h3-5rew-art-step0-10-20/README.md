# MiniMax-H3 T2VA, five-reward set (VideoAlign 1.0, HPSv3 1.5, AudioBox .5, CLAP 1.0, DeSync 1.0): DiffusionART step 0 · 10 · 20

Validation prompts 0–15 (README VidProM test split), same seed per row set, 512×768×121f, 40 steps; frames at 0/33/67/100 %.
Validation reward over 128 prompts: 9.63 → 12.45 → 12.68 (paired 10→20 Δ +.23 ± .13, 75/128 up). Components at 0/10/20: VideoAlign
1.22/1.31/1.37, HPSv3 6.71/9.29/9.35, AudioBox .387/.407/.433, CLAP .582/.605/.616, DeSync .730/.839/.909 — HPSv3 saturates after step 10,
the audio/sync rewards carry the later gain. Pixel statistics on these 16 clips vs step 0: **luma ×1.56 (16/16 up), contrast ×1.26,
saturation ×0.73, sharpness ×4.63 (16/16 up), motion ×0.67** — no black/frozen collapse, but a strong, monotone drift toward bright,
desaturated, over-sharpened frames with a third less motion.

![00](00.jpg)

![01](01.jpg)

![02](02.jpg)

![03](03.jpg)

![04](04.jpg)

![05](05.jpg)

![06](06.jpg)

![07](07.jpg)

![08](08.jpg)

![09](09.jpg)

![10](10.jpg)

![11](11.jpg)

![12](12.jpg)

![13](13.jpg)

![14](14.jpg)

![15](15.jpg)

