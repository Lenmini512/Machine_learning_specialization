# Machine Learning Specialization | Learning Portfolio 🤖📚

> Personal study notes, original Python implementations, and experiments developed while learning from **Andrew Ng's Machine Learning Specialization** by DeepLearning.AI and Stanford Online on Coursera.

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-Numerical_Computing-013243?logo=numpy&logoColor=white) ![Status](https://img.shields.io/badge/Status-Learning_in_Progress-yellow)

## About this repository

I am documenting my journey through machine learning by working through course concepts and reinforcing them with **my own code, visualizations, and small experiments**. I use VS Code and Jupyter to connect mathematical intuition with hands-on implementation.

My goal is to understand *why* an algorithm works, not only how to run it. As I progress, I plan to explore applications in engineering, signal processing and biomedical data.

## Learning approach

1. **Study:** Learn the concept and its mathematical foundations.
2. **Practice:** Work through course demonstrations and permitted practice exercises in my local environment.
3. **Implement:** Rebuild important ideas using original Python and NumPy code.
4. **Experiment:** Change assumptions, parameters and datasets; visualize the effects.
5. **Reflect:** Record observations, limitations and next steps.

## Course roadmap

| Course | Topics | Notes | Status |
|---|---|---|---|
| [Course 1: Supervised Machine Learning](Course-1-Supervised-Learning/README.md) | Linear regression, gradient descent, classification | [Course 1 notes](notes/Course-1-Supervised-Learning/) | Completed |
| [Course 2: Advanced Learning Algorithms](Course-2-Advanced-Learning-Algorithms/README.md) | Neural networks, model development, decision trees | [Course 2 notes](notes/Course-2-Advanced-Learning-Algorithms/) | In progress |
| [Course 3: Unsupervised Learning, Recommenders & RL](Course-3-Unsupervised-Recommenders-RL/README.md) | Clustering, anomaly detection, recommendation, reinforcement learning | [Course 3 notes](notes/Course-3-Unsupervised-Recommenders-RL/) | Not started |

## Week-by-week navigation

### Supervised Machine Learning: Regression and Classification

- [Week 1: Introduction to Machine Learning](Course-1-Supervised-Learning/Week-1-Introduction-and-Linear-Regression/README.md)
- [Week 2: Regression with Multiple Input Variables](Course-1-Supervised-Learning/Week-2-Multiple-Linear-Regression/README.md)
- [Week 3: Classification](Course-1-Supervised-Learning/Week-3-Classification/README.md)

### Advanced Learning Algorithms

- [Week 1: Neural Networks](Course-2-Advanced-Learning-Algorithms/Week-1-Neural-Networks/README.md)
- [Week 2: Neural Network Training](Course-2-Advanced-Learning-Algorithms/Week-2-Neural-Network-Training/README.md)
- [Week 3: Advice for Applying Machine Learning](Course-2-Advanced-Learning-Algorithms/Week-3-ML-Development/README.md)
- [Week 4: Decision Trees](Course-2-Advanced-Learning-Algorithms/Week-4-Decision-Trees/README.md)

### Unsupervised Learning, Recommenders, Reinforcement Learning

- [Week 1: Unsupervised Learning](Course-3-Unsupervised-Recommenders-RL/Week-1-Unsupervised-Learning/README.md)
- [Week 2: Recommender Systems](Course-3-Unsupervised-Recommenders-RL/Week-2-Recommender-Systems/README.md)
- [Week 3: Reinforcement Learning](Course-3-Unsupervised-Recommenders-RL/Week-3-Reinforcement-Learning/README.md)

## Repository layout

```text
Machine_learning_specialization/
├── Course-1-Supervised-Learning/
├── Course-2-Advanced-Learning-Algorithms/
├── Course-3-Unsupervised-Recommenders-RL/
├── notes/                    # My own study notes in PDF or Markdown
├── datasets/                 # Datasets permitted to be shared
├── mini-projects/            # Independent applications
├── assets/                   # My plots and diagrams
├── requirements.txt
└── README.md
```

## Personal notes 📄

I keep my own explanations, mathematical derivations and revision summaries under [`notes/`](notes/README.md). Notes are added as each topic is studied.

> **Note:** Only original notes and materials I have permission to redistribute are published here.

## Tools

Python • NumPy • Pandas • Matplotlib • Jupyter Notebook • scikit-learn • TensorFlow (when needed) • VS Code • Git

## Reproduce my experiments

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Open any `.ipynb` notebook in VS Code and choose the project's `.venv` kernel. Some lecture-provided helper files may be required for local course demonstrations and are not necessarily included in this public repository.

## Mini-projects

See [`mini-projects/`](mini-projects/) for independent projects as they are developed. Project names and folders represent ideas or works in progress, not claims of completion.

## References and attribution

- [Machine Learning Specialization — Coursera](https://www.coursera.org/specializations/machine-learning-introduction)
- [Machine Learning Specialization — DeepLearning.AI](https://www.deeplearning.ai/specializations/machine-learning)

**Disclaimer:** This is an independently maintained personal learning repository, not an official Coursera or DeepLearning.AI repository. I do not publish restricted graded assignments, course solutions or proprietary supporting materials.
