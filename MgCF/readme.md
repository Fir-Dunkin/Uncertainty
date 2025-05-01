# Wisdom via Multiple Perspectives: A Multi-granularity Clusters Fusion Approach for Fault Diagnosis with Noisy Labels

![image](https://github.com/user-attachments/assets/0df7f5e6-09a8-484d-bc23-48a3b984f7eb)

The overview of MgCF: For datasets with noisy labels, MgCF starts by using a feature encoder (the backbone of deep diagnostic model) to map the input sample signals into a high-dimensional feature space. 
    Then, MgCF dynamically constructs feature clusters with specific granularity, based on the estimated noise intensity in the dataset, for each feature representation.
    Next, MgCF integrates the information of all observation labels corresponding to the features within each cluster on a hypersphere to estimate the category membership of current sample. 
    Subsequently, the category membership is defuzzified to generate a predicted label with a category membership degree, and this predicted label, combined with the original observation label, forms multi-granularity labels for self-guided learning, thereby completing the supervised training of the diagnostic model.

```
@ARTICLE{Dunkin2025Wisdom,
  author={Dunkin, Fir and Li, Xinde and Wu, Guoliang and Hu, Chuanfei and Yu, Le and Lu, Xiaoyan and Ge, Shuzhi Sam},
  journal={IEEE/ASME Transactions on Mechatronics}, 
  title={Wisdom via Multiple Perspectives: A Multigranularity Clusters Fusion Approach for Fault Diagnosis With Noisy Labels}, 
  year={2025},
  volume={Early Access},
  number={},
  pages={1-11},
  keywords={Noise measurement;Training;Noise;Fault diagnosis;Annotations;Adaptation models;Accuracy;Mechatronics;Predictive models;Feature extraction;Feature fusion;fuzzy inference;granular computing;learning with noisy labels;time-series classification},
  doi={10.1109/TMECH.2025.3558839}}
```
