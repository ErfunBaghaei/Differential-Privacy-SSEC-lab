# Differential-Privacy-SSEC-lab
The elegant theoretical foundation and impressive empirical performance of Brainstorming Generative Adversarial Network (BGAN) and its variations have recently captivated researchers. These models hold great promise in scenarios where data availability is limited. However, a common challenge faced by BGANs is that the learned generative distribution tends to overly focus on the training data points, leading to easy memorization of the samples due to the complexity of deep networks. This issue becomes particularly worrisome when dealing with private or sensitive data, like patient medical records, as the concentration of the distribution may inadvertently reveal critical patient information. To tackle this concern, we propose a privacy approach called Differential Privacy. We introduce carefully crafted noise to gradients during the learning procedure to achieve differential privacy in BGANs. We provide rigorous proof of privacy guarantees and present comprehensive empirical evidence that demonstrates our method's ability to generate high-quality data points while ensuring a reasonable level of privacy.
