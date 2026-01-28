Globally, prostate cancer is the second most
widespread cancer among men, underscoring the significance
of accurate prostate segmentation in early detection, staging,
and treatment planning. Automating the segmentation of these
areas remains difficult because of subtle tissue contrasts and
irregular boundaries.Most importantly, segmenting peripheral
zone is always a challenge due to most of the cancers develop near
peripheral zone. This study presents a SwinUNet architecture,
designed with a focus on peripheral, to address the challenges of
prostate zonal segmentation. Our approach features several novel
components, including patch extraction centered on peripheral
to guarantee comprehensive visualisation of prostate anatomy,
and self-attention mechanisms in the bottleneck layer to capture
vital long-range spatial connections necessary for identifying zone
boundaries. We also applied volume calibration post-processing
to correct systematic estimation biases without compromising
segmentation accuracy. Previous research has frequently en
countered challenges in attaining high accuracy for prostate
zonal segmentation, with the peripheral zone being especially
troublesome. The majority of existing U-Net based models obtain
Dice scores in the peripheral zone that are below 80%, which
hampers their clinical utility. Existing methods are substantially
outperformed by our approach. The proposed peripheral-centric
SwinUNet surpassed traditional U-Net architectures with a pe
ripheral zone Dice coefficient of 0.82 (82%) and a transition
zone Dice coefficient of 0.94 (94%). Analysis of the qualitative
data showed improved definition of boundaries and a decrease
in incorrect positive areas, and attention visualization confirmed
that the model concentrates effectively on anatomically relevant
structures during segmentation. These results establish a new
benchmark for prostate zonal segmentation, with direct clinical
implications for enhanced cancer risk assessment, more accurate
treatment planning, and automated diagnostic workflows in
prostate healthcare.
