#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import

import numpy as np
import itertools
import ast
import math
import scipy
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(0)

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import datasets
from sklearn import svm
from sklearn import linear_model
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize

np.random.seed(0)

sigI = np.array([[1.0, 0.0j], [0.0j, 1.0]])
sigX = np.array([[0.0j, 1.0], [1.0, 0.0j]])
sigY = np.array([[0.0j, -1.0j], [1.0j, 0.0j]])
sigZ = np.array([[1.0, 0.0j], [0.0j, -1.0]])

N = 50

def kron(ls):
    A = ls[0]
    for X in ls[1:]:
        A = np.kron(A, X)
    return A

def generate_all_zero_state():
    return [np.array([[1.0, 0.0j], [0.0j, 0.0]]) for i in range(N)]

def generate_all_one_state():
    return [np.array([[0.0, 0.0j], [0.0j, 1.0]]) for i in range(N)]

def generate_half_half_state():
    return [np.array([[0.0, 0.0j], [0.0j, 1.0]]) if i < N/2 else np.array([[1.0, 0.0j], [0.0j, 0.0]]) for i in range(N)]

def generate_neel_state():
    return [np.array([[0.0, 0.0j], [0.0j, 1.0]]) if i % 2 == 0 else np.array([[1.0, 0.0j], [0.0j, 0.0]]) for i in range(N)]

def generate_all_plus_state():
    return [np.array([[0.5, 0.5], [0.5, 0.5+0.0j]]) for i in range(N)]

def generate_random_product_state():
    list_rhoi = []
    for i in range(N):
        v = np.random.normal(size=3)
        v /= np.linalg.norm(v)
        rhoi = sigI / 2.0 + (v[0] * sigX / 2.0) + (v[1] * sigY / 2.0) + (v[2] * sigZ / 2.0)
        list_rhoi.append(rhoi)
    return list_rhoi

def twobytwo_to_Pauli(list_rhoi):
    list_rhoi_new = []
    for rhoi in list_rhoi:
        list_rhoi_new.append(np.trace(np.matmul(sigX, rhoi)).real)
        list_rhoi_new.append(np.trace(np.matmul(sigY, rhoi)).real)
        list_rhoi_new.append(np.trace(np.matmul(sigZ, rhoi)).real)
    return list_rhoi_new

def get_RDM_in_Pauli(list_rhoi, k):
    feat_vec = []
    for i in range(N-k+1):
        for list_P in itertools.product([-1, 0, 1, 2], repeat=k):
            val = 1.0
            for c, P in enumerate(list_P):
                if P == -1: continue
                val *= list_rhoi[(3*(i+c))+P]
            assert(np.abs(val.imag) < 1e-7)
            feat_vec.append(val.real)
    return feat_vec

# Train a sparsity-enforcing ML model
def train_sparse_ML(all_states, all_values, test_size = 0.25, random_seed = 0):
    list_of_score = []
    list_of_clf = []
    list_of_bestk = []

    for pos in range(0, len(all_values[0])):
        print("Pos:", pos)

        def twobytwo_to_Pauli(list_rhoi):
            list_rhoi_new = []
            for rhoi in list_rhoi:
                list_rhoi_new.append(np.trace(np.matmul(sigX, rhoi)).real)
                list_rhoi_new.append(np.trace(np.matmul(sigY, rhoi)).real)
                list_rhoi_new.append(np.trace(np.matmul(sigZ, rhoi)).real)
            return list_rhoi_new

        def get_RDM_in_Pauli(list_rhoi, k):
            feat_vec = []
            for i in range(N-k+1):
                for list_P in itertools.product([-1, 0, 1, 2], repeat=k):
                    val = 1.0
                    for c, P in enumerate(list_P):
                        if P == -1: continue
                        val *= list_rhoi[(3*(i+c))+P]
                    assert(np.abs(val.imag) < 1e-7)
                    feat_vec.append(val.real)
            return feat_vec

        best_cv_score = 999.0
        best_clf = None
        best_k = None

        _, test_idx, _, _ = train_test_split(range(len(all_states)), range(len(all_states)), test_size=test_size, random_state=random_seed)

        for k in [1, 2, 3, 4]:
            print("Validate k =", k)
            X, y_true, y_noisy = [], [], []

            for data in zip(all_states, all_values):
                X.append(get_RDM_in_Pauli(data[0], k))
                y_true.append(data[1][pos])
                y_noisy.append((2 * np.random.binomial(500, (data[1][pos]+1)/2, 1)[0] / 500) - 1)

            X = np.array(X)
            y_true = np.array(y_true)
            y_noisy = np.array(y_noisy)

            X_train, X_test, y_train, y_test = train_test_split(X, y_noisy, test_size=test_size, random_state=random_seed)

            ML_method = lambda Cx : linear_model.Lasso(alpha=Cx)
            # ML_method = lambda Cx: linear_model.Ridge(alpha=Cx)

            for alpha in [2**(-15), 2**(-14), 2**(-13), 2**(-12), 2**(-11), 2**(-10), 2**(-9), 2**(-8), 2**(-7), 2**(-6), 2**(-5), 2**(-4), 2**(-3)]:
                score = -np.mean(cross_val_score(ML_method(alpha), X_train, y_train, cv=2, scoring="neg_root_mean_squared_error"))
                print(score)
                if best_cv_score > score:
                    clf = ML_method(alpha).fit(X_train, y_train)

                    best_cv_score = score
                    best_clf = clf
                    best_k = k

                    y_pred = clf.predict(X_test)
                    test_score = np.linalg.norm(y_pred - y_true[test_idx]) / (len(y_pred) ** 0.5)

        print("Scores:", best_cv_score, test_score)
        list_of_score.append(test_score)
        list_of_clf.append(best_clf)
        list_of_bestk.append(best_k)
        
    return list_of_score, list_of_clf, list_of_bestk

# Train a sparsity-enforcing ML model
def train_sparse_ML_transformed(all_X_list, all_values, test_size = 0.25, random_seed = 0):
    list_of_score = []
    list_of_clf = []
    list_of_bestk = []

    for pos in range(0, len(all_values[0])):
        print("Pos:", pos)

        best_cv_score = 999.0
        best_clf = None
        best_k = None

        _, test_idx, _, _ = train_test_split(range(len(all_states)), range(len(all_states)), test_size=test_size, random_state=random_seed)

        for k in [1, 2, 3, 4]:
            print("Validate k =", k)

            X = all_X_list[k-1]
            
            y_true, y_noisy = [], []
            for data in all_values:
                y_true.append(data[pos])
                y_noisy.append((2 * np.random.binomial(500, (data[pos]+1)/2, 1)[0] / 500) - 1)
            y_true = np.array(y_true)
            y_noisy = np.array(y_noisy)

            X_train, X_test, y_train, y_test = train_test_split(X, y_noisy, test_size=test_size, random_state=random_seed)

            ML_method = lambda Cx : linear_model.Lasso(alpha=Cx)
            # ML_method = lambda Cx: linear_model.Ridge(alpha=Cx)

            for alpha in [2**(-15), 2**(-14), 2**(-13), 2**(-12), 2**(-11), 2**(-10), 2**(-9), 2**(-8), 2**(-7), 2**(-6), 2**(-5), 2**(-4), 2**(-3)]:
                score = -np.mean(cross_val_score(ML_method(alpha), X_train, y_train, cv=2, scoring="neg_root_mean_squared_error"))
                print(score)
                if best_cv_score > score:
                    clf = ML_method(alpha).fit(X_train, y_train)

                    best_cv_score = score
                    best_clf = clf
                    best_k = k

                    y_pred = clf.predict(X_test)
                    test_score = np.linalg.norm(y_pred - y_true[test_idx]) / (len(y_pred) ** 0.5)

        print("Scores:", best_cv_score, test_score)
        list_of_score.append(test_score)
        list_of_clf.append(best_clf)
        list_of_bestk.append(best_k)
        
    return list_of_score, list_of_clf, list_of_bestk

def transform_states(all_states):
    all_X_list = []
    
    for k in [1, 2, 3, 4]:
        X = []
        for data in all_states:
            X.append(get_RDM_in_Pauli(data, k))
        all_X_list.append(np.array(X))

    return all_X_list


# # Error scaling with training set size

# In[2]:


N = 50
all_data_training_set_scaling = []

# XY model with homogeneous field

all_states = []
all_values = []

with open("50spins-oneZ-allt-homogeneous/states.txt") as f:
    for line in f:
        all_states.append(ast.literal_eval(line))

with open("50spins-oneZ-allt-homogeneous/values.txt") as f:
    for line in f:
        all_values.append([ast.literal_eval(line)[6]])

all_X_list = transform_states(all_states)
        
for test_size in [0.999, 0.997, 0.99, 0.97, 0.9, 0.7, 0.1]: # 10, 30, 100, 300, 1000, 3000, 9000 # 注释中对应的就是训练集的大小, 相当于训练集越大, test set 就越小. 
    for seed in range(10):
        list_of_score_AA, _, _ = train_sparse_ML_transformed(all_X_list, all_values, test_size=test_size, random_seed=seed)
        all_data_training_set_scaling.append([(1 - test_size) * 10000, 'XY (Homogeneous)', seed, list_of_score_AA[0]])
print(all_data_training_set_scaling)

# Ising model with homogeneous field

all_states = []
all_values = []

with open("50spins-oneZ-allt-homogeneous-Ising/states.txt") as f:
    for line in f:
        all_states.append(ast.literal_eval(line))

with open("50spins-oneZ-allt-homogeneous-Ising/values.txt") as f:
    for line in f:
        all_values.append([ast.literal_eval(line)[6]])

all_X_list = transform_states(all_states)
        
for test_size in [0.999, 0.997, 0.99, 0.97, 0.9, 0.7, 0.1]: # 10, 30, 100, 300, 1000, 3000, 9000
    for seed in range(10):
        list_of_score_AA, _, _ = train_sparse_ML_transformed(all_X_list, all_values, test_size=test_size, random_seed=seed)
        all_data_training_set_scaling.append([(1 - test_size) * 10000, 'Ising (Homogeneous)', seed, list_of_score_AA[0]])
print(all_data_training_set_scaling)

# XY model with disordered field
    
all_states = []
all_values = []

with open("50spins-oneZ-allt-disorder/states.txt") as f:
    for line in f:
        all_states.append(ast.literal_eval(line))

with open("50spins-oneZ-allt-disorder/values.txt") as f:
    for line in f:
        all_values.append([ast.literal_eval(line)[6]])

all_X_list = transform_states(all_states)
        
for test_size in [0.999, 0.997, 0.99, 0.97, 0.9, 0.7, 0.1]: # 10, 30, 100, 300, 1000, 3000, 9000
    for seed in range(10):
        list_of_score_AA, _, _ = train_sparse_ML_transformed(all_X_list, all_values, test_size=test_size, random_seed=seed)
        all_data_training_set_scaling.append([(1 - test_size) * 10000, 'XY (Disorder)', seed, list_of_score_AA[0]])
print(all_data_training_set_scaling)

# Ising model with disordered field

all_states = []
all_values = []

with open("50spins-oneZ-allt-disorder-Ising/states.txt") as f:
    for line in f:
        all_states.append(ast.literal_eval(line))

with open("50spins-oneZ-allt-disorder-Ising/values.txt") as f:
    for line in f:
        all_values.append([ast.literal_eval(line)[6]])

all_X_list = transform_states(all_states)
        
for test_size in [0.999, 0.997, 0.99, 0.97, 0.9, 0.7, 0.1]: # 10, 30, 100, 300, 1000, 3000, 9000
    for seed in range(10):
        list_of_score_AA, _, _ = train_sparse_ML_transformed(all_X_list, all_values, test_size=test_size, random_seed=seed)
        all_data_training_set_scaling.append([(1 - test_size) * 10000, 'Ising (Disorder)', seed, list_of_score_AA[0]])
print(all_data_training_set_scaling)


# In[ ]:


sns.set_style("ticks")
plt.figure(figsize=(2.9, 2.6))

N_list = np.array([10, 30, 100, 300, 1000, 3000, 9000])
Nalpha_list = 0.5 * N_list ** (-1/3)
plt.plot(N_list, Nalpha_list, label=r"$0.5 \, / \, N^{1/3}$", linestyle=':', linewidth=1, c='k', alpha=0.8)
    
df = pd.DataFrame(data=all_data_training_set_scaling, columns=["Training set size", "Model", "Seed", "Error"])
ax = sns.lineplot(data=df[df['Model'] == "Ising (Homogeneous)"], x="Training set size", y="Error", label="Ising (Homo.)", markers=True, marker='D', markersize=7, c='#A6CFE3', alpha=0.7)
ax = sns.lineplot(data=df[df['Model'] == "XY (Homogeneous)"], x="Training set size", y="Error", label="XY (Homo.)", markers=True, marker='o', markersize=7, c='#1F78B4', alpha=0.8)

# plt.plot(list_time, list_of_score_HXY, label='XY (Homogeneous)', linestyle='-', linewidth=1, marker='o', markersize=4, c='#AEDA87', alpha=0.8)
# plt.plot(list_time, list_of_score_HIsing, label='Ising (Homogeneous)', linestyle='-', linewidth=1, marker='D', markersize=4, c='#329D2B', alpha=0.8)
# plt.plot(list_time, list_of_score_DXY, label='XY (Disorder)', linestyle='--', linewidth=1, marker='s', markersize=4, c='#A6CFE3', alpha=0.8)
# plt.plot(list_time, list_of_score_DIsing, label='Ising (Disorder)', linestyle='--', linewidth=1, marker='^', markersize=5, c='#1F78B4', alpha=0.7)

ax.set_xscale('log')
plt.xticks([10, 100, 1000, 10000])
plt.yticks([0.0, 0.05, 0.1, 0.15, 0.2, 0.25])
plt.ylim(0.0, 0.275)

plt.legend()

plt.xlabel(r'Training set size $(N)$', fontsize=10);
plt.ylabel(r'Prediction error (RMSE)', fontsize=10);
plt.tick_params(axis='both', which='major', labelsize=10)

plt.tight_layout()
plt.savefig('SizeN-homogeneous.pdf', dpi=900)


# In[ ]:


sns.set_style("ticks")
plt.figure(figsize=(2.9, 2.6))

N_list = np.array([10, 30, 100, 300, 1000, 3000, 9000])
Nalpha_list = 1.5 * N_list ** (-1/2)
plt.plot(N_list, Nalpha_list, label=r"$1.5 \, / \, \sqrt{N}$", linestyle=':', linewidth=1, c='k', alpha=0.8)

df = pd.DataFrame(data=all_data_training_set_scaling, columns=["Training set size", "Model", "Seed", "Error"])
ax = sns.lineplot(data=df[df['Model'] == "Ising (Disorder)"], x="Training set size", y="Error", label="Ising (Disord.)", linestyle='--', markers=True, marker='^', markersize=7, c='#AEDA87', alpha=0.8)
ax = sns.lineplot(data=df[df['Model'] == "XY (Disorder)"], x="Training set size", y="Error", label="XY (Disord.)", linestyle='--', markers=True, marker='s', markersize=7, c='#329D2B', alpha=0.8)

ax.set_xscale('log')
plt.xticks([10, 100, 1000, 10000])
plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
plt.ylim(0, 0.55)

plt.legend()

plt.xlabel(r'Training set size $(N)$', fontsize=10);
plt.ylabel(r'Prediction error (RMSE)', fontsize=10);
plt.tick_params(axis='both', which='major', labelsize=10)

plt.tight_layout()
# plt.savefig('SizeN-disorder.pdf', d`pi=900)


# # Error scaling with evolution time

# In[ ]:


# XY model with disordered field

N = 50
all_states = []
all_values = []

with open("50spins-oneZ-allt-disorder/states.txt") as f:
    for line in f:
        all_states.append(ast.literal_eval(line))

with open("50spins-oneZ-allt-disorder/values.txt") as f:
    for line in f:
        all_values.append(ast.literal_eval(line))

list_of_score_DXY, list_of_clf_DXY, list_of_bestk_DXY = train_sparse_ML(all_states, all_values)

# XY model with homogeneous field

N = 50
all_states = []
all_values = []

with open("50spins-oneZ-allt-homogeneous/states.txt") as f:
    for line in f:
        all_states.append(ast.literal_eval(line))

with open("50spins-oneZ-allt-homogeneous/values.txt") as f:
    for line in f:
        all_values.append(ast.literal_eval(line))

list_of_score_HXY, list_of_clf_HXY, list_of_bestk_HXY = train_sparse_ML(all_states, all_values)

# Ising model with disordered field

N = 50
all_states = []
all_values = []

with open("50spins-oneZ-allt-disorder-Ising/states.txt") as f:
    for line in f:
        all_states.append(ast.literal_eval(line))

with open("50spins-oneZ-allt-disorder-Ising/values.txt") as f:
    for line in f:
        all_values.append(ast.literal_eval(line))

list_of_score_DIsing, list_of_clf_DIsing, list_of_bestk_DIsing = train_sparse_ML(all_states, all_values)

# Ising model with homogeneous field

N = 50
all_states = []
all_values = []

with open("50spins-oneZ-allt-homogeneous-Ising/states.txt") as f:
    for line in f:
        all_states.append(ast.literal_eval(line))

with open("50spins-oneZ-allt-homogeneous-Ising/values.txt") as f:
    for line in f:
        all_values.append(ast.literal_eval(line))

list_of_score_HIsing, list_of_clf_HIsing, list_of_bestk_HIsing = train_sparse_ML(all_states, all_values)


# Print out the test score
print(list_of_score_DXY)
print(list_of_score_HXY)
print(list_of_score_DIsing)
print(list_of_score_HIsing)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'widget')
sns.set_style("ticks")

plot1 = plt.figure(figsize=(3.2, 2.55))

list_of_score_DXY = [0.007915236382114544, 0.04947091105957198, 0.04697752770941811, 0.05191098963262961, 0.03764809304216626, 0.011861222560826395, 0.04830875023735673]
list_of_score_HXY = [0.013256966670524326, 0.03721494102988028, 0.02897388745290167, 0.029803642600065405, 0.026695180719768098, 0.027710837095259566, 0.03173716275137415]
list_of_score_DIsing = [0.018531582079668242, 0.05244811900644047, 0.060618771949532635, 0.032888046159887455, 0.04902412952128371, 0.06912464490177862, 0.054527279125304154]
list_of_score_HIsing = [0.012453854027517147, 0.046725475639309094, 0.033806992302953455, 0.0331672725880195, 0.037965462774408515, 0.032736439079639836, 0.035616811446614396]

list_time = [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0]
plt.plot(list_time, list_of_score_DIsing, label='Ising (Disordered)', linestyle='--', linewidth=1, marker='^', markersize=5, c='#AEDA87', alpha=0.8)
plt.plot(list_time, list_of_score_DXY, label='XY (Disordered)', linestyle='--', linewidth=1, marker='s', markersize=4, c='#329D2B', alpha=0.8)
plt.plot(list_time, list_of_score_HIsing, label='Ising (Homogeneous)', linestyle='-', linewidth=1, marker='D', markersize=4, c='#A6CFE3', alpha=0.7)
plt.plot(list_time, list_of_score_HXY, label='XY (Homogeneous)', linestyle='-', linewidth=1, marker='o', markersize=4, c='#1F78B4', alpha=0.8)


plt.xscale('log')
plt.ylim(-0.0, 0.3)
plt.xlabel("Evolution time")
plt.ylabel("Prediction error (RMSE)")
plt.xticks(list_time)
plt.legend()

plt.tight_layout()
plt.show()
plt.savefig('Evo-time.pdf', dpi=900)


# # Error scaling with system size

# In[ ]:


all_score = []
all_std = []

for N in [10, 20, 30, 40, 50]:
    all_states = []
    all_values = []

    with open("Sys-" + str(N) + "spins-oneZ-allt-homogeneous/states.txt") as f:
        for line in f:
            all_states.append(ast.literal_eval(line))

    with open("Sys-" + str(N) + "spins-oneZ-allt-homogeneous/values.txt") as f:
        for line in f:
            all_values.append(ast.literal_eval(line))

    list_of_score_N, list_of_clf_N, list_of_bestk_N = train_sparse_ML(all_states, all_values)
    
    for v in range(len(all_values[0])):
        all_std.append(np.std(np.array(all_values)[:, v]))
    
    all_score.append(list_of_score_N)

print(all_score)
print(all_std)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'widget')
sns.set_style("ticks")

plot1 = plt.figure(figsize=(3.2, 2.55))

list_time = [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0]

all_score = [[0.01322497269545254, 0.04479558921588049, 0.04292973910856739, 0.029978702246647934, 0.052000697259225445, 0.04494183912184762, 0.048572455160393914], [0.013238151411245697, 0.034490569038691764, 0.030699445164745386, 0.04863870949641657, 0.03528956178937412, 0.04077764991409794, 0.03918693649968483], [0.013327195712980354, 0.03543493750607781, 0.030076289465591825, 0.029608329621221716, 0.032373592811570505, 0.03404159565397256, 0.030068691637193418], [0.012905884564335715, 0.038844056207704196, 0.034592378302616404, 0.03352689516506794, 0.03299109407866409, 0.03360457955846986, 0.03341170415568, 0.03634060633516722], [0.013598927845442432, 0.03730917782077512, 0.028135506651756566, 0.030428638086740904, 0.026699431994580268, 0.027228809474526966, 0.031829182030938936]]

normal_score = []
cnt = 0
for n in range(5):
    ls = []
    for x in range(len(all_score[0])):
        ls.append(all_score[n][x])# / all_std[cnt] * all_std[1])
        cnt += 1
    normal_score.append(ls)
        
plt.plot(list_time, normal_score[0], label='n = 10 spins', linestyle='-', linewidth=1, marker='o', markersize=4, c=sns.color_palette("Blues")[1], alpha=0.9)
plt.plot(list_time, normal_score[1], label='n = 20 spins', linestyle='-', linewidth=1, marker='o', markersize=4, c=sns.color_palette("Blues")[2], alpha=0.9)
plt.plot(list_time, normal_score[2], label='n = 30 spins', linestyle='-', linewidth=1, marker='o', markersize=4, c=sns.color_palette("Blues")[3], alpha=0.9)
plt.plot(list_time, normal_score[3], label='n = 40 spins', linestyle='-', linewidth=1, marker='o', markersize=4, c=sns.color_palette("Blues")[4], alpha=0.9)
plt.plot(list_time, normal_score[4], label='n = 50 spins', linestyle='-', linewidth=1, marker='o', markersize=4, c=sns.color_palette("Blues")[5], alpha=0.9)


plt.xscale('log')
plt.ylim(-0.0, 0.3)
plt.xlabel("Evolution time")
plt.ylabel("Prediction error (RMSE)")
plt.xticks(list_time)
plt.legend()

plt.tight_layout()
plt.show()
plt.savefig('System-size.pdf', dpi=900)


# # Visualize predictions

# In[ ]:


N = 50
all_states = []
all_values = []

with open("50spins-allZ-many-t-homogeneous/states.txt") as f:
    for line in f:
        all_states.append(ast.literal_eval(line))

with open("50spins-allZ-many-t-homogeneous/values.txt") as f:
    for line in f:
        all_values.append(ast.literal_eval(line))

list_of_score_HXY_vis, list_of_clf_HXY_vis, list_of_bestk_HXY_vis = train_sparse_ML(all_states, all_values)


# In[ ]:


# Pickle dump
pickle.dump( list_of_score_HXY_vis, open( "list_of_score_HXY_vis.p", "wb" ) )
pickle.dump( list_of_clf_HXY_vis, open( "list_of_clf_HXY_vis.p", "wb" ) )
pickle.dump( list_of_bestk_HXY_vis, open( "list_of_bestk_HXY_vis.p", "wb" ) )


# In[ ]:


# Pickle load
list_of_score_HXY_vis_pkl = pickle.load( open( "list_of_score_HXY_vis.p", "rb" ) )
list_of_clf_HXY_vis_pkl = pickle.load( open( "list_of_clf_HXY_vis.p", "rb" ) )
list_of_bestk_HXY_vis_pkl = pickle.load( open( "list_of_bestk_HXY_vis.p", "rb" ) )


# In[ ]:


# Pickle load (just to be safe)
list_of_score_HXY_vis, list_of_clf_HXY_vis, list_of_bestk_HXY_vis = list_of_score_HXY_vis_pkl, list_of_clf_HXY_vis_pkl, list_of_bestk_HXY_vis_pkl


# In[ ]:


# Visualize half-half state

N = 50

def get_feat_vec_half_half(k):
    feat_vec = []
    for i in range(N-k+1):
        for list_P in itertools.product([-1, 0, 1, 2], repeat=k):
            val = 1.0
            
            for c, P in enumerate(list_P):
                if P == -1: continue
                if P != 2: val = 0.0
                
                if (i+c) < N / 2: val *= -1
                else: val *= +1
                    
            assert(np.abs(val.imag) < 1e-7)
            feat_vec.append(val.real)
    return feat_vec

evo_time_list = [0, 1, 4, 10, 1000, 1000000]
true_ans = np.array([-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-0.999988,-0.999222,-0.972817,-0.585528,0.585528,0.972817,0.999222,0.999988,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-0.999998,-0.999966,-0.999505,-0.994686,-0.959792,-0.801725,-0.43163,-0.166451,-0.157728,0.157728,0.166451,0.43163,0.801725,0.959792,0.994686,0.999505,0.999966,0.999998,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-0.999999,-0.999995,-0.999954,-0.999668,-0.997989,-0.989958,-0.959642,-0.873541,-0.703182,-0.501119,-0.407192,-0.406774,-0.297204,-0.200754,-0.193937,-0.0642642,-0.0604844,0.0604844,0.0642642,0.193937,0.200754,0.297204,0.406774,0.407192,0.501119,0.703182,0.873541,0.959642,0.989958,0.997989,0.999668,0.999954,0.999995,0.999999,1,1,1,1,1,1,1,1,0.223538,0.0704778,0.0547808,0.137909,-0.109168,0.0870547,-0.0704109,0.255476,-0.132473,0.0208235,0.0104076,-0.210476,0.00149826,-0.0734874,-0.0874279,0.326401,0.0580026,-0.146868,-0.00298605,-0.0620534,-0.292064,-0.337162,0.0267581,0.126973,0.0217619,-0.0217619,-0.126973,-0.0267581,0.337162,0.292064,0.0620534,0.00298605,0.146868,-0.0580026,-0.326401,0.0874279,0.0734874,-0.00149826,0.210476,-0.0104076,-0.0208235,0.132473,-0.255476,0.0704109,-0.0870547,0.109168,-0.137909,-0.0547808,-0.0704778,-0.223538,-0.122712,-0.239076,0.121464,-0.116011,-0.195256,-0.1169,-0.0462604,0.231811,0.0477476,-0.121555,0.194615,0.0496414,0.0103274,-0.0615695,0.220349,-0.198141,0.0778494,0.041073,0.261736,0.111922,-0.222931,0.0772192,-0.0577005,0.108646,0.147718,-0.147718,-0.108646,0.0577005,-0.0772192,0.222931,-0.111922,-0.261736,-0.041073,-0.0778494,0.198141,-0.220349,0.0615695,-0.0103274,-0.0496414,-0.194615,0.121555,-0.0477476,-0.231811,0.0462604,0.1169,0.195256,0.116011,-0.121464,0.239076,0.122712,]).reshape(-1, N)

print(len(list_of_score_HXY_vis))
for i in range(6):
    f = plt.figure(figsize=(4.63, 2.3))
    
    pred_ans = [list_of_clf_HXY_vis[t].predict(np.array([get_feat_vec_half_half(list_of_bestk_HXY_vis[t])]))[0] for t in range(50*i, 50*(i+1))]
    
    plt.plot(range(1, 51), true_ans[i, :], 'D', linestyle='-', linewidth=0.9, label='Truth (Exact)', markersize=3.5, c='#1F78B4', alpha=1.0)
    plt.plot(range(1, 51), pred_ans, 'D', mfc='none', label='ML Prediction', markersize=3.5, c='#929292', alpha=1.0)

    plt.xlabel(r"$i$-th spin in a 50-spin 1D XY chain", fontsize=12)
    plt.ylabel(r"$\langle \, Z_i(t) \, \rangle$".format(evo_time_list[i]), fontsize=12)
    plt.legend(loc='upper left')
    plt.xticks([1, 10, 20, 30, 40, 50])
    plt.ylim([-1.2, 1.2])
#     plt.title(r"$| 111 \ldots 000 \rangle$" + r" at time $t = {}$".format(evo_time_list[i]))

    plt.tight_layout()
    plt.savefig('111000-allZ-{}.pdf'.format(i), dpi=900)
    


# In[ ]:


# Visualize entangled Neel state

N = 50

def get_feat_vec_GHZ_Neel(k):
    assert(k < N / 2)
    feat_vec = []
    for i in range(N-k+1):
        for list_P in itertools.product([-1, 0, 1, 2], repeat=k):
            val1 = 1.0
            
            for c, P in enumerate(list_P):
                if P == -1: continue
                if P != 2: val1 = 0.0
                
                if (i+c) % 2 == 1: val1 *= -1
                else: val1 *= +1

            val2 = 1.0
            
            for c, P in enumerate(list_P):
                if P == -1: continue
                if P != 2: val2 = 0.0
                
                if (i+c) >= N / 2: val2 *= -1
                    
            val = (val1 + val2) / 2

            assert(np.abs(val.imag) < 1e-7)
            feat_vec.append(val.real)
    return feat_vec

evo_time_list = [0, 1, 4, 10, 1000, 1000000]
true_ans = np.array([1,-3.33067e-16,1,-3.33067e-16,1,0,1,2.22045e-16,1,2.22045e-16,1,2.22045e-16,1,0,1,-1.11022e-16,1,0,1,-2.22045e-16,1,-1.11022e-16,1,0,1,-1,-6.66134e-16,-1,-2.22045e-16,-1,0,-1,-4.44089e-16,-1,-2.22045e-16,-1,-1.11022e-16,-1,-1.11022e-16,-1,0,-1,2.22045e-16,-1,2.22045e-16,-1,2.22045e-16,-1,2.22045e-16,-1,0.788362,0.405052,0.612547,0.388066,0.611946,0.388055,0.611945,0.388055,0.611945,0.388055,0.611945,0.388055,0.611945,0.388055,0.611945,0.388055,0.611945,0.388055,0.611945,0.388055,0.611945,0.388048,0.611556,0.374463,0.404709,-0.404709,-0.374463,-0.611556,-0.388048,-0.611945,-0.388055,-0.611945,-0.388055,-0.611945,-0.388055,-0.611945,-0.388055,-0.611945,-0.388055,-0.611945,-0.388055,-0.611945,-0.388055,-0.611945,-0.388055,-0.611946,-0.388066,-0.612547,-0.405052,-0.788362,0.52933,0.361496,0.754613,0.525902,0.616209,0.418987,0.586335,0.414214,0.585828,0.414175,0.585825,0.414175,0.585825,0.414175,0.585825,0.414175,0.585824,0.414157,0.585578,0.411518,0.565721,0.315037,0.30164,-0.0025997,0.164689,-0.164689,0.0025997,-0.30164,-0.315037,-0.565721,-0.411518,-0.585578,-0.414157,-0.585824,-0.414175,-0.585825,-0.414175,-0.585825,-0.414175,-0.585825,-0.414175,-0.585828,-0.414214,-0.586335,-0.418987,-0.616209,-0.525902,-0.754613,-0.361496,-0.52933,0.503342,0.481823,0.555969,0.379553,0.676754,0.356992,0.510313,0.489078,0.709057,0.498859,0.617281,0.426286,0.584769,0.411879,0.563395,0.353266,0.435104,0.167047,0.287108,0.119875,0.232115,0.0168645,0.180481,-0.0513803,0.113755,-0.113755,0.0513803,-0.180481,-0.0168645,-0.232115,-0.119875,-0.287108,-0.167047,-0.435104,-0.353266,-0.563395,-0.411879,-0.584769,-0.426286,-0.617281,-0.498859,-0.709057,-0.489078,-0.510313,-0.356992,-0.676754,-0.379553,-0.555969,-0.481823,-0.503342,0.0641132,-0.156605,0.0212648,-0.211451,0.151916,-0.189812,0.176076,-0.276748,0.165638,-0.124994,0.182763,-0.109042,0.187432,-0.002042,0.159606,-0.263859,0.097639,-0.00387948,0.12612,-0.103179,0.263876,0.102068,0.119708,-0.18066,0.101832,-0.101832,0.18066,-0.119708,-0.102068,-0.263876,0.103179,-0.12612,0.00387948,-0.097639,0.263859,-0.159606,0.002042,-0.187432,0.109042,-0.182763,0.124994,-0.165638,0.276748,-0.176076,0.189812,-0.151916,0.211451,-0.0212648,0.156605,-0.0641132,0.15872,0.084307,0.00893069,-0.0615725,0.223252,-0.0967308,0.182103,-0.244717,-0.0320069,-0.122955,-0.0108487,-0.115121,0.0397093,-0.0642493,-0.0126599,-0.0731859,0.0733459,-0.0360828,-0.0679437,-0.190291,0.248513,-0.218302,0.125698,-0.218771,-0.016088,0.016088,0.218771,-0.125698,0.218302,-0.248513,0.190291,0.0679437,0.0360828,-0.0733459,0.0731859,0.0126599,0.0642493,-0.0397093,0.115121,0.0108487,0.122955,0.0320069,0.244717,-0.182103,0.0967308,-0.223252,0.0615725,-0.00893069,-0.084307,-0.15872,]).reshape(-1, N)

print(len(list_of_score_HXY_vis))
for i in range(6):
    f = plt.figure(figsize=(4.63, 2.3))
    
    pred_ans = [list_of_clf_HXY_vis[t].predict(np.array([get_feat_vec_GHZ_Neel(list_of_bestk_HXY_vis[t])]))[0] for t in range(50*i, 50*(i+1))]
    
    plt.plot(range(1, 51), true_ans[i, :], 'D', linestyle='-', linewidth=0.9, label='Truth (Exact)', markersize=3.5, c='#1F78B4', alpha=1.0)
    plt.plot(range(1, 51), pred_ans, 'D', mfc='none', label='ML Prediction', markersize=3.5, c='#929292', alpha=1.0)

    plt.xlabel(r"$i$-th spin in a 50-spin 1D XY chain", fontsize=12)
    plt.ylabel(r"$\langle \, Z_i(t) \, \rangle$".format(evo_time_list[i]), fontsize=12)
    plt.legend()
    plt.xticks([1, 10, 20, 30, 40, 50])
    plt.ylim([-1.2, 1.2])
#     plt.title(r"$\frac{1}{\sqrt{2}} | uu \ldots dd \rangle + \frac{1}{\sqrt{2}}| udud \ldots \rangle$" + r" at time $t = {}$".format(evo_time_list[i]))

    plt.tight_layout()
    plt.savefig('Neel-allZ-{}.pdf'.format(i), dpi=900)
    


# In[ ]:


# Visualize W state

N = 50
scale = 3.0

def get_feat_vec_Entangled(k):
    assert(k < N / 2)
    feat_vec = []
    for i in range(N-k+1):
        for list_P in itertools.product([-1, 0, 1, 2], repeat=k):
            val1 = 1.0
            
            for c, P in enumerate(list_P):
                if P == -1: continue
                
                if (i+c) < 20:
                    if P != 0: val1 = 0.0
                    else: val1 *= -1
                else:
                    if P == 1: val1 = 0.0
                        
                    theta = np.pi/4 + (scale * np.pi * (i+c - 20)) / 30
                    if P == 0:
                        val1 *= (np.cos(theta) * np.sin(theta) + np.sin(theta) * np.cos(theta))
                    if P == 2:
                        val1 *= (np.cos(theta) * np.cos(theta) - np.sin(theta) * np.sin(theta))

            val2 = 1.0
            
            for c, P in enumerate(list_P):
                if P == -1: continue
                
                if (i+c) < 20:
                    if P != 0: val2 = 0.0
                    else: val2 *= +1
                else:
                    if P == 1: val2 = 0.0
                        
                    theta = np.pi/4 + (scale * np.pi * (i+c - 20)) / 30
                    if P == 0:
                        val2 *= (np.cos(theta) * np.sin(theta) + np.sin(theta) * np.cos(theta))
                    if P == 2:
                        val2 *= (np.cos(theta) * np.cos(theta) - np.sin(theta) * np.sin(theta))
                    
            val = (val1 + val2) / 2

            assert(np.abs(val.imag) < 1e-7)
            feat_vec.append(val.real)
    return feat_vec

evo_time_list = [0, 1, 4, 10, 1000, 1000000]
ls = [-1.33227e-15,-9.99201e-16,-5.55112e-16,-1.11022e-15,-2.22045e-16,-6.66134e-16,-3.33067e-16,-5.55112e-16,-4.44089e-16,-4.44089e-16,-4.44089e-16,-1.11022e-16,-9.99201e-16,-7.77156e-16,-7.77156e-16,-8.88178e-16,-6.66134e-16,-8.88178e-16,-1.55431e-15,-1.11022e-15,-3.33067e-16,-0.587785,-0.951057,-0.951057,-0.587785,-7.77156e-16,0.587785,0.951057,0.951057,0.587785,2.22045e-16,-0.587785,-0.951057,-0.951057,-0.587785,-1.44329e-15,0.587785,0.951057,0.951057,0.587785,1.55431e-15,-0.587785,-0.951057,-0.951057,-0.587785,-2.77556e-15,0.587785,0.951057,0.951057,0.587785,1.23693e-09,-1.18242e-09,-2.22045e-16,-5.0973e-11,-3.51097e-12,-6.66134e-16,-2.86438e-14,2.88658e-15,-1.9984e-15,-2.10942e-15,0,1.55431e-15,1.9984e-15,-3.51941e-14,-1.69564e-12,-3.3124e-10,-4.65665e-08,-4.55288e-06,-0.000282417,-0.00967345,-0.141464,-0.480152,-0.90726,-0.906982,-0.470479,2.80743e-09,0.470479,0.906978,0.906978,0.470479,3.13666e-09,-0.470479,-0.906978,-0.906978,-0.470479,3.9968e-15,0.470479,0.906978,0.906978,0.470479,-3.69665e-10,-0.470479,-0.906978,-0.906978,-0.470479,-4.16475e-05,0.469544,0.908928,0.894072,0.643797,-3.52094e-10,1.77685e-09,1.55431e-15,2.74368e-09,-3.52092e-09,6.66134e-16,-5.66338e-10,-7.85829e-11,-7.66942e-13,-3.3301e-11,-1.0525e-09,-3.00228e-08,-7.00873e-07,-1.30575e-05,-0.00018838,-0.0020226,-0.0153012,-0.0753679,-0.214593,-0.30259,-0.263781,-0.339562,-0.232529,-0.0933044,-0.0522733,-0.00202258,0.0367837,0.0179235,0.0179358,0.036972,1.21569e-08,-0.0369721,-0.0179365,-0.0179365,-0.0369721,-1.10911e-13,0.0369721,0.0179365,0.0179365,0.036972,-2.2049e-06,-0.0370048,-0.0182925,-0.0205336,-0.0475969,-0.00960933,0.127346,0.318042,0.491566,0.680018,-2.51953e-10,-9.52393e-10,-1.56492e-09,-1.82514e-08,-2.27022e-07,-2.15299e-06,-1.82034e-05,-0.000131701,-0.000801645,-0.00402792,-0.0162984,-0.0514086,-0.120913,-0.199239,-0.211137,-0.132791,-0.0508541,0.029253,0.112162,0.0800971,0.0207406,-0.0760861,-0.159016,-0.241925,-0.207037,-0.132791,-0.0549536,0.0719396,0.150265,0.104775,-0.0162984,-0.160211,-0.27198,-0.271312,-0.15622,-0.000118119,0.155599,0.268883,0.264755,0.146866,0.0132259,-0.0423814,0.0361288,0.19502,0.313248,0.421725,0.432211,0.251384,-0.176082,-0.314243,0.04488,0.0475582,0.0705712,-0.00391528,-0.128828,0.177717,-0.0454162,0.139205,-0.0470074,0.149673,-0.17846,-0.129918,-0.106554,0.0771985,-0.0862278,0.00140352,0.0220758,-0.0527357,-0.0166446,0.0392979,0.0734952,0.00896325,-0.0619655,-0.2245,-0.0345697,-0.130262,0.0895834,0.157284,0.243013,-0.0903524,0.146047,0.0580688,-0.0891992,-0.204532,-0.141727,-0.0685656,0.142277,0.186667,0.106245,0.00729997,-0.175042,0.0430485,-0.131919,-0.186464,0.0403929,0.106612,0.0208504,0.00851356,0.115719,0.0111436,0.128465,0.0364219,-0.0424433,-0.258732,-0.118112,-0.235543,-0.171896,-0.048835,0.0897999,0.00470972,0.0190959,-0.041822,-0.0149158,0.0273092,-0.145442,0.0261904,0.0175673,-0.0475926,-0.0272873,0.125373,0.0124977,0.252649,0.112097,0.0788469,-0.0107116,-0.0795201,-0.155343,-0.128473,-0.0933882,-0.0890382,-0.0239084,-0.150032,0.0653788,0.120773,0.156574,0.106836,0.0995081,-0.00397812,-0.105693,-0.0100067,0.093005,0.101245,0.0836636,0.0823051,0.186772,0.113701,0.0844316,0.0146368,0.0187623,-0.255901,]
true_ans = np.array(ls).reshape(-1, N)

print(len(list_of_score_HXY_vis))
for i in range(6):
    f = plt.figure(figsize=(4.63, 2.3))
    
    pred_ans = [list_of_clf_HXY_vis[t].predict(np.array([get_feat_vec_W(list_of_bestk_HXY_vis[t])]))[0] for t in range(50*i, 50*(i+1))]
    
    plt.plot(range(1, 51), true_ans[i, :], 'D', linestyle='-', linewidth=0.9, label='Truth (Exact)', markersize=3.5, c='#1F78B4', alpha=1.0)
    plt.plot(range(1, 51), pred_ans, 'D', mfc='none', label='ML Prediction', markersize=3.5, c='#929292', alpha=1.0)

    plt.xlabel(r"$i$-th spin in a 50-spin 1D XY chain", fontsize=12)
    plt.ylabel(r"$\langle \, Z_i(t) \, \rangle$".format(evo_time_list[i]), fontsize=12)
    plt.legend()
    plt.xticks([1, 10, 20, 30, 40, 50])
    plt.ylim([-1.05, 1.05])

    plt.tight_layout()
    plt.savefig('Entangled-allZ-{}.pdf'.format(i), dpi=900)
    


# In[ ]:


# Visualize Oscillating state

N = 50

scale = 4.0
oscillating_len = 32
entangled_len = N - oscillating_len

def get_feat_vec_W(k):
    assert(k < N / 2)
    feat_vec = []
    for i in range(N-k+1):
        for list_P in itertools.product([-1, 0, 1, 2], repeat=k):
            val1 = 1.0
            
            for c, P in enumerate(list_P):
                if P == -1: continue
                
                if (i+c) < entangled_len:
                    if P != 2: val1 = 0.0
                    else: val1 *= -1
                else:
                    if P == 1: val1 = 0.0
                        
                    theta = np.pi/4 + (scale * np.pi * (i+c - entangled_len)) / oscillating_len
                    if P == 0:
                        val1 *= (np.cos(theta) * np.sin(theta) + np.sin(theta) * np.cos(theta))
                    if P == 2:
                        val1 *= (np.cos(theta) * np.cos(theta) - np.sin(theta) * np.sin(theta))

            val2 = 1.0
            
            for c, P in enumerate(list_P):
                if P == -1: continue
                
                if (i+c) < entangled_len:
                    if P != 2: val2 = 0.0
                    else: val2 *= +1
                else:
                    if P == 1: val2 = 0.0
                        
                    theta = np.pi/4 + (scale * np.pi * (i+c - entangled_len)) / oscillating_len
                    if P == 0:
                        val2 *= (np.cos(theta) * np.sin(theta) + np.sin(theta) * np.cos(theta))
                    if P == 2:
                        val2 *= (np.cos(theta) * np.cos(theta) - np.sin(theta) * np.sin(theta))
                    
            val = (val1 + val2) / 2

            assert(np.abs(val.imag) < 1e-7)
            feat_vec.append(val.real)
    return feat_vec

evo_time_list = [0, 1, 4, 10, 1000, 1000000]
ls = [-5.55112e-16,-6.66134e-16,-3.33067e-16,-6.66134e-16,0,-3.33067e-16,-1.11022e-16,-1.11022e-16,-4.44089e-16,-1.11022e-16,-1.11022e-16,-1.11022e-16,-4.44089e-16,-3.33067e-16,-1.11022e-16,-4.44089e-16,-3.33067e-16,-3.33067e-16,-6.66134e-16,-0.707107,-1,-0.707107,-3.33067e-16,0.707107,1,0.707107,-6.66134e-16,-0.707107,-1,-0.707107,-4.44089e-16,0.707107,1,0.707107,-3.33067e-16,-0.707107,-1,-0.707107,-2.77556e-15,0.707107,1,0.707107,-1.11022e-15,-0.707107,-1,-0.707107,-2.66454e-15,0.707107,1,0.707107,0,-5.55112e-16,-1.22125e-15,-9.99201e-16,0,-3.33067e-16,2.22045e-16,0,-2.10942e-15,-1.77636e-15,1.33227e-15,-5.10703e-15,-1.5925e-12,-3.11422e-10,-4.44776e-08,-4.38862e-06,-0.000275565,-0.00963552,-0.147044,-0.56874,-0.96253,-0.559109,-4.16168e-08,0.559104,0.962255,0.559104,1.33227e-15,-0.559104,-0.962255,-0.559104,-2.01598e-09,0.559104,0.962255,0.559104,-2.01598e-09,-0.559104,-0.962255,-0.559104,3.33067e-15,0.559104,0.962255,0.559104,2.74466e-09,-0.559105,-0.962256,-0.559161,-0.000999399,0.562674,0.968682,0.717039,2.66454e-15,1.55431e-15,-8.21565e-15,-4.77396e-15,1.77636e-15,-1.64313e-14,-7.62834e-13,-2.86574e-11,-9.83163e-10,-2.81194e-08,-6.57331e-07,-1.22692e-05,-0.000177372,-0.00190871,-0.0144726,-0.0713878,-0.202568,-0.277234,-0.210429,-0.224771,-0.0298034,-0.0189252,-0.0144726,-0.0543713,-0.172942,-0.0524749,-6.57372e-07,0.0524626,0.172765,0.0524626,1.60704e-09,-0.0524626,-0.172765,-0.0524626,1.60771e-09,0.0524626,0.172765,0.0524626,-1.26962e-08,-0.0524629,-0.17277,-0.0525289,-0.000621166,0.0487576,0.162028,0.059554,0.128932,0.251019,0.354465,0.34189,-1.64891e-09,-1.80879e-08,-2.03576e-07,-1.98558e-06,-1.67752e-05,-0.000121095,-0.000735011,-0.00367672,-0.0147676,-0.0459711,-0.10537,-0.163749,-0.146579,-0.0407221,0.0427528,0.0699782,0.0522136,-0.0655041,-0.137071,-0.190526,-0.148734,-0.0550436,0.0427527,0.0842996,0.0543686,-0.0387276,-0.10537,-0.170993,-0.215715,-0.128698,-0.000735046,0.124901,0.20093,0.125012,-5.35602e-05,-0.12533,-0.202373,-0.129992,-0.0115816,0.114316,0.23493,0.28248,0.307373,0.260138,0.204349,0.172876,0.00395937,-0.145266,-0.282008,0.297803,-0.214572,-0.184999,-0.0858761,-0.0189614,0.0970766,0.159462,0.184308,-0.0371108,0.0431076,-0.0620992,-0.0533646,0.0711563,-0.105234,-0.0412968,-0.00540601,-0.0321334,0.00893819,0.0564909,-0.0228622,-0.0270342,-0.0390996,0.0786328,0.0274031,-0.0230434,0.188251,0.0373823,-0.114784,-0.118867,-0.135593,-0.00545273,0.0490424,0.0261038,0.0789864,0.149492,-0.0646498,0.0359345,-0.106258,-0.136215,0.117361,-0.0348563,0.152954,0.0624063,0.0612586,0.158723,0.0652991,0.0293106,-0.00661968,-0.178976,-0.0625079,-0.0212084,-0.107264,-0.0704725,0.193965,0.108294,0.0717049,0.156469,-0.0648497,-0.072299,-0.139204,0.0490899,0.0910658,0.0349663,-0.0954868,-0.172838,-0.141793,0.00729624,0.0761089,0.139629,0.126779,-0.165521,-0.0712546,-0.0862275,-0.077053,0.123855,-0.057867,-0.00401261,-0.139977,-0.171911,-0.152037,0.225867,0.210818,0.0540612,0.112161,-0.0380994,-0.084963,-0.202104,-0.0479781,-0.171457,0.122403,0.0217878,0.175638,-0.0748639,-0.102267,0.172358,-0.0882058,0.0305625,0.0908711,0.0497088,0.0588678,0.0956785,]
true_ans = np.array(ls).reshape(-1, N)

print(len(list_of_score_HXY_vis))
for i in range(6):
    f = plt.figure(figsize=(4.63, 2.3))
    
    pred_ans = [list_of_clf_HXY_vis[t].predict(np.array([get_feat_vec_W(list_of_bestk_HXY_vis[t])]))[0] for t in range(50*i, 50*(i+1))]
    
    plt.plot(range(1, 51), true_ans[i, :], 'D', linestyle='-', linewidth=0.9, label='Truth (Exact)', markersize=3.5, c='#1F78B4', alpha=1.0)
    plt.plot(range(1, 51), pred_ans, 'D', mfc='none', label='ML Prediction', markersize=3.5, c='#929292', alpha=1.0)

    plt.xlabel(r"$i$-th spin in a 50-spin 1D XY chain", fontsize=12)
    plt.ylabel(r"$\langle \, Z_i(t) \, \rangle$".format(evo_time_list[i]), fontsize=12)
    plt.legend()
    plt.xticks([1, 10, 20, 30, 40, 50])
    plt.ylim([-1.05, 1.05])

    plt.tight_layout()
    plt.savefig('Entangled-4-allZ-{}.pdf'.format(i), dpi=900)
    


# In[ ]:


# Visualize Oscillating state

N = 50

scale = 0.25
oscillating_len = 25
entangled_len = N - oscillating_len

def get_feat_vec_W(k):
    assert(k < N / 2)
    feat_vec = []
    for i in range(N-k+1):
        for list_P in itertools.product([-1, 0, 1, 2], repeat=k):
            val1 = 1.0
            
            for c, P in enumerate(list_P):
                if P == -1: continue
                
                if (i+c) < entangled_len:
                    if P != 2: val1 = 0.0
                    else: val1 *= -1
                else:
                    if P == 1: val1 = 0.0
                        
                    theta = np.pi/4 + (scale * np.pi * (i+c - entangled_len))
                    if P == 0:
                        val1 *= (np.cos(theta) * np.sin(theta) + np.sin(theta) * np.cos(theta))
                    if P == 2:
                        val1 *= (np.cos(theta) * np.cos(theta) - np.sin(theta) * np.sin(theta))

            val2 = 1.0
            
            for c, P in enumerate(list_P):
                if P == -1: continue
                
                if (i+c) < entangled_len:
                    if P != 2: val2 = 0.0
                    else: val2 *= +1
                else:
                    if P == 1: val2 = 0.0
                        
                    theta = np.pi/4 + (scale * np.pi * (i+c - entangled_len))
                    if P == 0:
                        val2 *= (np.cos(theta) * np.sin(theta) + np.sin(theta) * np.cos(theta))
                    if P == 2:
                        val2 *= (np.cos(theta) * np.cos(theta) - np.sin(theta) * np.sin(theta))
                    
            val = (val1 + val2) / 2

            assert(np.abs(val.imag) < 1e-7)
            feat_vec.append(val.real)
    return feat_vec

evo_time_list = [0, 1, 4, 10, 1000, 1000000]
ls = [0,-1.11022e-16,2.22045e-16,-1.11022e-16,6.66134e-16,2.22045e-16,4.44089e-16,4.44089e-16,0,4.44089e-16,4.44089e-16,4.44089e-16,0,2.22045e-16,4.44089e-16,0,2.22045e-16,2.22045e-16,-4.44089e-16,0,4.44089e-16,0,4.44089e-16,2.22045e-16,2.22045e-16,6.66134e-16,-1,-2.22045e-16,1,4.44089e-16,-1,-4.44089e-16,1,2.22045e-16,-1,-2.33147e-15,1,-8.88178e-16,-1,-2.66454e-15,1,-3.33067e-16,-1,-2.88658e-15,1,2.22045e-16,-1,-2.33147e-15,1,0,6.66134e-16,0,-6.66134e-16,-4.44089e-16,6.66134e-16,2.22045e-16,6.66134e-16,4.44089e-16,-1.55431e-15,-9.99201e-16,1.77636e-15,1.55431e-15,-7.77156e-16,-7.77156e-16,0,-1.22125e-15,6.66134e-16,1.77636e-15,2.66454e-15,-3.14748e-13,-6.3195e-11,-1.05254e-08,-1.24763e-06,-9.81398e-05,-0.0045927,-0.105622,-0.774579,-9.81411e-05,0.769985,-1.05219e-08,-0.769987,1.35209e-09,0.769987,1.35238e-09,-0.769987,2.22045e-15,0.769987,-1.35238e-09,-0.769987,-1.35239e-09,0.769987,-7.99361e-15,-0.769987,1.147e-09,0.769987,-2.37629e-06,-0.770101,-0.00201803,0.779289,0.107839,3.33067e-15,1.9984e-15,-7.77156e-15,-3.33067e-15,4.44089e-15,-8.88178e-16,1.9984e-15,4.88498e-15,-8.65974e-15,-7.88258e-15,9.54792e-15,7.99361e-15,-1.22125e-15,-6.55032e-15,-3.15847e-12,-1.15585e-10,-3.68859e-09,-9.51021e-08,-1.98227e-06,-3.2461e-05,-0.000403052,-0.00361448,-0.0218106,-0.079066,-0.139189,-0.0699091,-0.022901,-0.079066,-0.138098,-0.00361448,0.115885,-3.24485e-05,-0.11629,-8.31988e-08,0.116288,-1.17724e-10,-0.116288,-1.19193e-08,0.116288,-3.2007e-08,-0.116288,-8.65716e-06,0.116165,-0.00125065,-0.124702,-0.0301418,0.102946,0.211788,0.2284,0.0549484,2.22045e-16,0,-5.10703e-15,-2.5091e-14,6.66134e-16,-4.29434e-13,-6.75471e-12,-1.01429e-10,-1.41485e-09,-1.66839e-08,-1.73902e-07,-1.57113e-06,-1.21947e-05,-8.00841e-05,-0.000436652,-0.00192916,-0.00668157,-0.0173003,-0.0311014,-0.0342591,-0.0198314,-0.0173505,-0.0420583,-0.0419752,-0.0202404,-0.0225374,-0.0421339,-0.0419752,-0.0201647,-0.0173504,-0.041725,-0.0342592,-0.00920796,-0.0173018,-0.0285877,-0.00202041,0.020899,-0.00288813,-0.0331993,-0.0346213,-0.0519068,-0.0824906,0.0274971,0.326942,0.432659,0.0577022,-0.187023,0.0133466,0.0532551,-0.0296511,0.167654,0.235669,0.059428,-0.0647698,0.00977301,0.0359311,-0.142181,-0.268125,-0.08796,0.12132,0.123859,-0.0365826,-0.0825699,0.0243864,-0.0286014,-0.217735,-0.182816,-0.0233282,-0.0754973,-0.174467,-0.0535967,-0.0515959,-0.148926,-0.169623,-0.0281208,0.0987317,-0.0348927,-0.180714,0.0084423,0.319679,0.266015,0.0292598,0.0332248,0.16749,0.0365815,-0.137161,-0.0667869,0.118105,0.148816,-0.103928,-0.179366,-0.0127684,-0.00616234,-0.0335855,0.0411815,0.253788,0.18819,0.0495114,0.0343125,0.0205118,0.0190869,0.158465,0.0807447,-0.0769012,-0.104517,0.0260456,0.0361047,-0.23948,-0.242446,0.0959721,0.142571,-0.201348,-0.216818,0.245565,0.317963,-0.102833,-0.232337,0.08118,0.084799,-0.105552,-0.00880639,0.0923163,-0.089633,-0.2577,-0.0768566,0.0812138,0.150611,0.0129071,-0.0417149,0.134226,0.0912652,-0.00312032,-0.0284457,0.0284919,0.0180958,-0.0224571,0.00262732,0.103439,0.0929488,0.0665141,0.16849,0.127639,-0.00581823,-0.160224,-0.136392,-0.0476056,0.00417718,-0.00822495,-0.0256998,-0.0285305,]
true_ans = np.array(ls).reshape(-1, N)

print(len(list_of_score_HXY_vis))
for i in range(6):
    f = plt.figure(figsize=(4.63, 2.3))
    
    pred_ans = [list_of_clf_HXY_vis[t].predict(np.array([get_feat_vec_W(list_of_bestk_HXY_vis[t])]))[0] for t in range(50*i, 50*(i+1))]
    
    plt.plot(range(1, 51), true_ans[i, :], 'D', linestyle='-', linewidth=0.9, label='Truth (Exact)', markersize=3.5, c='#1F78B4', alpha=1.0)
    plt.plot(range(1, 51), pred_ans, 'D', mfc='none', label='ML Prediction', markersize=3.5, c='#929292', alpha=1.0)

    plt.xlabel(r"$i$-th spin in a 50-spin 1D XY chain", fontsize=12)
    plt.ylabel(r"$\langle \, Z_i(t) \, \rangle$".format(evo_time_list[i]), fontsize=12)
    plt.legend(loc='upper left')
    plt.xticks([1, 10, 20, 30, 40, 50])
    plt.ylim([-1.08, 1.08])

    plt.tight_layout()
    plt.savefig('Entangled-8-allZ-{}.pdf'.format(i), dpi=900)
    

