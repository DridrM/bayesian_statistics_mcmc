# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 18:19:08 2022

@author: MONCOMBH
"""

"""Création d'un script pour s'entrainer à l'échantillonage préférentiel"""

#Librairies

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from math import floor
from math import log10


"""Etape 1 : Définition de la loi à priori et de la loi de vraissemblance"""

#Choix des paramètres de la loi à priori (loi bêta)
alpha = float(input('Paramètre alpha = '))
beta = float(input('Paramètre beta = '))

#Plot de la loi beta
xb = np.linspace(0, 1, 400)
yb = scipy.stats.beta.pdf(xb, alpha, beta)

plt.figure()
plt.plot(xb, yb, color = 'green', label = f'Loi beta\n alpha = {alpha}\n beta = {beta}')
plt.grid()
plt.title('Loi à priori')
plt.xlabel('x')
plt.ylabel('Densité de x')
plt.legend()
plt.show()

#Choix des paramètres de la loi de vraissemblance (loi normale)
mu_v = float(input('Moyenne vraissemblance (variable aléatoire par la suite) = '))
sigma_v = float(input('Ecart-type vraissemblance = '))

#Plot de la loi normale
xv = np.linspace(mu_v - 4*sigma_v, mu_v + 4*sigma_v, 400)
yv = scipy.stats.norm.pdf(xv, mu_v, sigma_v)

plt.figure()
plt.plot(xv, yv, color = 'blue', label = f'Loi normale\n mu = {mu_v}\n sigma = {sigma_v}')
plt.grid()
plt.title('Loi de vraissemblance')
plt.xlabel('x')
plt.ylabel('Densité de x')
plt.legend()
plt.show()

"""Etape 2 : Choix la loi auxilliaire"""

#Choix des paramètres de la loi auxilliaire
mu_a = float(input('Moyenne loi auxilliaire = '))
sigma_a = float(input('Ecart-type loi auxilliaire = '))

#Choix du facteur multiplicatif loi auxilliaire
#m = float(input('Facteur multiplicatif = '))

#Plot de la distribution jointe et de la loi auxilliaire
xa = np.linspace(mu_a - 4*sigma_a, mu_a + 4*sigma_a, 400)
ya = scipy.stats.norm.pdf(xv, mu_a, sigma_a)

plt.figure()
plt.plot(xb, yv*yb, color = 'red', label = 'beta x normal')
plt.plot(xa, ya, color = 'coral', label = f'Loi normale\n mu = {mu_a}\n sigma = {sigma_a}')
plt.grid()
plt.title('Distribution jointe')
plt.xlabel('x')
plt.ylabel('Densité de x')
plt.legend()
plt.show()

"""Etape 3 : Algorithme de Metropolis-Hastings"""

#Choix nombre d'itérations de l'échantillonage
n = int(input('Nombre itérations echantillonage = '))
#Calcul nombre de classes pour l'histogramme
n_classes = int(floor(max(30, 10*log10(n))))
#Choix du burn-in
burn_in = int(floor(n/10))

#Choix de la moyenne observée
mu_obs = float(input('Moyenne observée = '))

#Définition moyenne aléatoire
def x_rand(mu, sigma) :
    x = np.random.normal(loc = mu, scale = sigma)
    x = float(x)
    return x

def y_rand(x, mu, sigma) :
    y = scipy.stats.norm.pdf(x, mu, sigma)
    y = float(y)
    return y

#Définition vraissemblance
def yv_rand(mu, x, sigma) :
    y = scipy.stats.norm.pdf(mu, x, sigma)
    y = float(y)
    return y

#Définition loi à priori
def yb_rand(x, alpha, beta) :
    y = scipy.stats.beta.pdf(x, alpha, beta)
    y = float(y)
    return y

#Algorithme type Markov Chain Monte Carlo : Echantillonneur de Metropolis-Hastings
xp = [mu_a]
for i in range(0, n+1) :
    #Obtention de la moyenne en tant que variable aléatoire
    theta = x_rand(xp[-1], sigma_a)
    #Echantillonage aléatoire selon la loi auxilliaire
    g = y_rand(theta, xp[-1], sigma_a)
    #Echantillonage aléatoire de compensation selon la loi auxilliaire
    g_comp = y_rand(xp[-1], theta, sigma_a)
    #Calcul distribution jointe -> Vraissemblance x loi à priori
    f = yv_rand(mu_obs, theta, sigma_v)*yb_rand(theta, alpha, beta)
    #Calcul distribution jointe de compensation -> Vraissemblance x loi à priori
    f_comp = yv_rand(mu_obs, xp[-1], sigma_v)*yb_rand(xp[-1], alpha, beta)
    #Calcul de la probabilité d'acceptation
    prob_accept = min(1, (f*g_comp)/(f_comp*g))
    
    if np.random.random() < prob_accept :
        xp.append(theta)
    
    else :
        xp.append(xp[-1])

#Déduction du burn-in
xp_final = xp[burn_in+1: ]

#Histogramme échantillonage : Résultat final
plt.figure()
plt.hist(xp_final, bins = n_classes, color = 'navy', density = True, label = 'Loi à postériori')
plt.plot(xb, yb, color = 'green', label = f'Loi beta\n alpha = {alpha}\n beta = {beta}')
plt.title('Loi à priori & loi à postériori')
plt.legend()
plt.show()
