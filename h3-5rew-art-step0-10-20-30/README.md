# MiniMax-H3 T2VA, five-reward set: DiffusionART step 0 · 10 · 20 · 30 (same prompt and seed per row set)

Validation reward over 128 prompts: 9.63 → 12.45 → 12.68 → 13.20 (paired 20→30 Δ +.52 ± .12, 83/128 up). Components at 0/10/20/30:
VideoAlign 1.22/1.31/1.37/1.43, HPSv3 6.71/9.29/9.35/9.76, AudioBox .387/.407/.433/.441, CLAP .582/.605/.616/.618, DeSync .730/.839/.909/.949.
Pixel statistics on these 16 clips vs step 0: **luma ×1.87 (16/16), contrast ×1.34, saturation ×0.62 (1/16 up), sharpness ×4.96 (16/16),
motion ×0.53**. The drift toward bright, desaturated, over-sharpened, half-as-moving frames continues monotonically while every reward
component still rises — none of the five scorers penalises it.

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

