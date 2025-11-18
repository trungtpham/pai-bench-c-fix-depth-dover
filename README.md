<p align="center">
  <img src="assets/physical-ai-bench-logo-20250923.png" alt="Physical AI Bench Logo" width="64%">
</p>

![Python Version](https://img.shields.io/badge/Python-3.10-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
[![Hugging Face - Leaderboard](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Leaderboard-orange)](https://huggingface.co/spaces/shi-labs/physical-ai-bench-leaderboard)
[![Hugging Face - Predict](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Predict-orange)](https://huggingface.co/datasets/shi-labs/physical-ai-bench-predict)
[![Hugging Face - Transfer](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transfer-orange)](https://huggingface.co/datasets/shi-labs/physical-ai-bench-transfer)
[![Hugging Face - Reason](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Reason-orange)](https://huggingface.co/datasets/shi-labs/physical-ai-bench-reason)
![Paper Coming Soon](https://img.shields.io/badge/Paper-Coming%20Soon-yellow)
![Georgia Tech](https://img.shields.io/badge/Affiliation-Georgia%20Tech-ad9e66)
![CMU](https://img.shields.io/badge/Affiliation-CMU-c41230)

## Introduction

Physical AI Bench (PAI-Bench) is a comprehensive benchmark suite for evaluating physical AI generation and understanding. PAI-Bench covers physical scenarios including autonomous vehicle (AV) driving, robotics, industry (smart space) and ego-centric everyday. PAI-Bench contains three subtasks:

- **Predict**: Evaluates world foundation models' ability to predict future states given current states and control signals
- **Transfer**: Focuses on world model generation capabilities with more complex control signals such as edges, segmentation masks, depth, etc.
- **Reason**: Evaluates understanding of physical scenes.

<p align="center">
  <img src="assets/physical-ai-bench-teaser-20250928.png" alt="Physical AI Bench Overview" width="100%">
</p>

## Datasets

| Tasks        | Data                                                                                                 | Usage                              |
| ------------ | ---------------------------------------------------------------------------------------------------- | ---------------------------------- |
| **Predict**  | [🤗 physical-ai-bench-predict](https://huggingface.co/datasets/shi-labs/physical-ai-bench-predict)   | [Link](./predict)  |
| **Transfer** | [🤗 physical-ai-bench-transfer](https://huggingface.co/datasets/shi-labs/physical-ai-bench-transfer) | [Link](./transfer) |
| **Reason**   | [🤗 physical-ai-bench-reason](https://huggingface.co/datasets/shi-labs/physical-ai-bench-reason)     | [Link](./reason)   |

## Leaderboard

Leaderboard is available on [🤗 physical-ai-bench-leaderboard](https://huggingface.co/spaces/shi-labs/physical-ai-bench-leaderboard).


## Citation

Paper is coming soon!

If you use Physical AI Bench in your research, please cite:

```bibtex
@misc{PAIBench2025,
  title={Physical AI Bench: A Comprehensive Benchmark for Physical AI Generation and Understanding},
  author={Fengzhe Zhou and Jiannan Huang and Jialuo Li and Deva Ramanan and Humphrey Shi},
  year={2025},
  url={https://github.com/SHI-Labs/physical-ai-bench}
}
```

## Acknowledgements

We would like to thank NVIDIA Research, especially the Cosmos team for their support which led to the creation of PAI-Bench. We also thank [Yin Cui](https://ycui.me/), [Jinwei Gu](https://www.gujinwei.org/), [Heng Wang](https://hengcv.github.io/), [Prithvijit Chattopadhyay](https://prithv1.xyz/), Andrew Z. Wang, [Imad El Hanafi](https://imadelh.gitlab.io/), and [Ming-Yu Liu](https://mingyuliu.net/) for their valuable feedback and collaboration that helped shaped the project. This research was supported in part by National Science Foundation under Award #2427478 - CAREER Program, and by National Science Foundation and the Institute of Education Sciences, U.S. Department of Education under Award #2229873 - National AI Institute for Exceptional Education. This project was also partially supported by cyberinfrastructure resources and services provided Georgia Institute of Technology.
