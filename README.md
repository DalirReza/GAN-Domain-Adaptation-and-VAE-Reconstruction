# GAN-Domain-Adaptation-and-VAE-Reconstruction

This repository contains the project files for the sixth project of the "Neural Networks and Deep Learning" course at the University of Tehran. This project explores two advanced topics in deep learning:

1.  **Unsupervised Domain Adaptation using Generative Adversarial Networks (GANs)** based on the PixelDA methodology.
2.  **Image Reconstruction and Generation using Variational Autoencoders (VAEs)**, with an implementation of the EndoVAE model for medical imaging.

> **Note:** The original assignment instructions and the final report are written in Persian. English versions can be provided upon request.

---
 
## Project Overview

This project is divided into two distinct parts, each tackling a different challenge with generative models.

### Part 1: Unsupervised Domain Adaptation with GANs (PixelDA)

The first problem addresses the "Domain Gap" or "Domain Shift" challenge, where a model trained on a source data distribution performs poorly on a target distribution with different visual characteristics.

* **Objective**: To train a digit classifier on the **MNIST** dataset and have it perform accurately on the **MNIST-M** (target domain) dataset, *without* using any labels from MNIST-M.
* **Methodology**: We implemented the **Pixel-Level Domain Adaptation (PixelDA)** framework. This model uses a GAN architecture with three core components:
    * A **Generator** that learns to translate images from the source domain (MNIST) to the style of the target domain (MNIST-M) while preserving the original digit.
    * A **Discriminator** that learns to distinguish between real MNIST-M images and the "fake" images created by the Generator.
    * A **Classifier** that is trained on both original MNIST images and the generated images to perform the final digit classification task.
* **Results**:
    * A baseline classifier trained only on MNIST achieved **~98%** accuracy on the MNIST test set but only **~62%** on the MNIST-M set, clearly demonstrating the domain gap.
    * After training the full PixelDA model, the classifier's accuracy on the MNIST-M target domain significantly increased to **~96%**.
    * This result shows that the model successfully bridged the domain gap by learning to adapt the visual style of the images, allowing the classifier to generalize effectively to the new domain.

### Part 2: Endoscopic Image Reconstruction with EndoVAE

The second problem explores the use of Variational Autoencoders for reconstructing and generating medical images, specifically for anomaly detection.

* **Objective**: To implement the **EndoVAE** model to learn the distribution of "normal" (healthy) endoscopic images. The primary goal is to evaluate its ability to reconstruct "abnormal" images (those containing polyps), which it has never seen during training.
* **Methodology**:
    * An EndoVAE model, consisting of an Encoder and a Decoder, was built and trained exclusively on a dataset of normal endoscopic images from the Kvasir dataset.
    * The model's loss function is a combination of a **Reconstruction Loss** (MSE was found to perform better than BCE) and the **KL Divergence**, which regularizes the latent space.
* **Results**:
    * **Qualitative**: When the model was tasked with reconstructing unseen polyp images, it produced blurry and distorted results. The anomalous regions (the polyps) were often removed or smoothed over, making the output resemble the normal images it was trained on. This is the desired behavior for an anomaly detection system.
    * **Quantitative**: The poor reconstruction quality was confirmed with low average metrics on 50 polyp images:
        * **PSNR**: ~17.56 dB
        * **SSIM**: ~0.45
    * This demonstrates that a high reconstruction error can be used as a signal to identify out-of-distribution (anomalous) data, validating the VAE's effectiveness for this task.

---
