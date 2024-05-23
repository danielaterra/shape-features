import numpy as np
import pandas as pd 
from math import sqrt
import os
import sys
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow 
from skimage import morphology, measure
from skimage.draw import polygon, polygon_perimeter
from scipy.spatial.distance import cdist
from scipy.stats import kurtosis
import pyefd
from pyefd import elliptic_fourier_descriptors, normalize_efd
from itertools import cycle
from random import randint
from random import sample
from collections import deque
import xgboost as xgb 
import csv
import cv2

#TODO: Fazer acesso do atributos da classe abaixo nas funções
class CRIC_images:
    def __init__(self):
        self.IMG_W = 1376
        self.IMG_H = 1020
        self.Bethesda_classes = {'Normal':0, 'ASC-US':1, 'LSIL':2, 'ASC-H':3,'HSIL':4, 'Invasive Carcinoma':5} 

###################################
## Extração de features:


def get_contour_cell_points(df_cell_masks, image_id, cell_id):
        img = CRIC_images()
        # 'show' values: NU - Nucleus, CY - Cytoplasm, CL          
        points_N = df_cell_masks.query('image_id == '+str(image_id) + ' and cell_id == '+str(cell_id))['nparray_points_segm_Nucleus'].values
        points_C = df_cell_masks.query('image_id == '+str(image_id) + ' and cell_id == '+str(cell_id))['nparray_points_segm_Cyto'].values    
        xN = points_N[0][:,0]
        yN = points_N[0][:,1]
        xC = points_C[0][:,0]
        yC = points_C[0][:,1] 
        return (xN, yN, xC, yC) 

### ---- Funções para curvatura ----##
def calc_curvatures(x, y, s=3):
    sigma = (np.real(x.shape[0])) * s/100
    x_smooth, gaus_f =  gaus_smooth_signal(x, sigma)
    y_smooth, gaus_f = gaus_smooth_signal(y, sigma)

    # Calcula x'(t),x''(t), y'(t), y''(t), a partir das transformadas:
    fft_x, freqs = fft_freqs(x_smooth)
    energy_x = total_energy(fft_x)
    dx, dx2 = derivatives(fft_x, freqs)

    fft_y, freqs = fft_freqs(y_smooth)
    energy_y = total_energy(fft_y)
    dy, dy2 = derivatives(fft_y, freqs)
    
    curvs = curvatures(dx, dx2, dy, dy2)
    m_curvs = np.abs(curvs)
    fft_curvs, freqs = fft_freqs(m_curvs)
    dK, d2K = derivatives(fft_curvs, freqs)
    zero_cross, max_points_mcurv = calc_max_points(dK, d2K)
     
    return (m_curvs, zero_cross, max_points_mcurv, calc_n_max_curvs(m_curvs, max_points_mcurv, n=5))

def calc_n_max_curvs(mcurvs, max_points_mcurv, n=5):
    l = []
    for i in max_points_mcurv:
        l.append(mcurvs[i])
    max_curvs = np.array(l)
    max_curvs.sort()
    return max_curvs[::-1][0:n]

# Funções curvatura:
def gaus_filter(n, n_median, sig):
   # n: numero de dados
   #n_median: posição para o vr maximo da curva de Gaus (ex. n/2) 
   #sigma: desvio padrão da gausiana
   gaus_f = np.zeros(n, dtype = complex)   
   for i in np.arange(n):
      gaus_f[i] = (1./(sig*(np.sqrt(2*np.pi))))*np.exp((-1*(i- n_median)**2)/(2*sig**2))
   return gaus_f
 
# Calcula o espectro de energia do vetor transformado antes da gaussina
def total_energy(fft_x):
   real = np.real(fft_x)**2
   imag = np.imag(fft_x)**2
   return (real + imag)

def calc_energy(tf):
    et = np.zeros(tf.shape[0], dtype = np.float64)
    for i in np.arange(tf.shape[0]):
        et[i] = (np.real(tf[i])**2) + (np.imag(tf[i])**2) 
    return et, np.sum(et)

def gaus_smooth_signal (x, sigma):
   n = x.shape[-1]
   gaus_f = gaus_filter(n, n/2, sigma)
   trans_gaus = np.fft.fft(gaus_f)
   trans_x=(np.fft.fft(x))
   trans_x = trans_x * trans_gaus
   itrans_x = np.fft.fftshift(np.fft.ifft(trans_x))
   return (itrans_x, gaus_f) 

# Calcula x'(t),x''(t), y'(t), y''(t), a partir de Fourier:
def derivatives(fft_x, freqs_x):   
   dx =  freqs_x*1j
   dx2 = dx**2
   dF = np.fft.fftshift(np.fft.ifft(fft_x*dx))
   d2F= np.fft.fftshift(np.fft.ifft(fft_x*dx2))
   return ((dF), (d2F))

# Calcula FFT e frequencias  
def fft_freqs(x):
   return (np.fft.fft(x), np.fft.fftfreq(x.shape[-1]))

# Calcula Curvatures
def curvatures(dx, dx2, dy, dy2):
   t = np.arange(dx.shape[-1])
   n= (dx[t]*dy2[t]) - (dy[t]*dx2[t])
   d= ((dx[t]**2) + (dy[t]**2))**(3.0/2)
   return (n/d) 
#----- 

# Busca pelos pontos máximos de curvatura
## Pontos de máximo - d2K/ds < 0:
def calc_max_points(dK, d2K):
    zero_cross= deque()
    max_points_mcurv= deque()
    d1 = np.real(dK)
    d2 = np.real(d2K)
    n = d1.shape[-1]
    for t in np.arange(n):
        if t < (n-1):
            if (np.sign(d1[t]) != np.sign(d1[t+1])):   
                zero_cross.appendleft(t)
                if (d2K[t] <0):
                    max_points_mcurv.appendleft(t)
                elif (d2[t+1] <0):
                    max_points_mcurv.appendleft(t+1)
    return (zero_cross, max_points_mcurv)
 
# Monta dataframe de features por célula (núcleo e citoplasma):
def list_cells(nucleos_csv, cyto_csv): 
   df_nucleos_full = pd.read_csv(nucleos_csv, header=0)
   df_cyto_full = pd.read_csv(cyto_csv, header = 0)
   
   # dataframe of unique cells (nucleos)
   df_nucleos = df_nucleos_full[['image_id', 'cell_id', 'image_filename']]
   df_nucleos = df_nucleos.sort_values(by=['image_id', 'cell_id']) 
   df_nucleos = df_nucleos.drop_duplicates(subset=['image_id', 'cell_id'], keep='first', inplace=False) 
    
   # dataframe of unique cells (cytoplams)
   df_cyto = df_cyto_full[['image_id', 'cell_id', 'image_filename']]
   df_cyto = df_cyto.sort_values(by=['image_id', 'cell_id']) 
   df_cyto= df_cyto.drop_duplicates(keep='first', inplace=False) 
    
   return (df_nucleos, df_cyto,df_nucleos_full, df_cyto_full)

# Calcula deslocamento relativo do núcleo dentro do citoplasma (baseado em versão \cite{Mariarputham2015}):
## Relativa ao Major axis do citoplasma
def nucleus_position(cent_N, cent_C, major_axis_C):
    d = np.sqrt((cent_N[0]-cent_C[0])**2 + (cent_N[1]-cent_C[1])**2)
    if d == 0:
        return 0
    else: 
        return (d/major_axis_C)   

## NÃO USADO - Para chamada ao metodo que approxima o contorno com a série de Fourier series, como descrito em (https://www.sci.utah.edu/~gerig/CS7960-S2010/handouts/Kuhl-Giardina-CGIP1982.pdf)
## Fonte: https://pyefd.readthedocs.io/en/latest/#second
def efd_feature(contour, n_coeffs):
    '''  contour: pontos de borda
         n_coeffs: nº de coeficientes da serie de fourier (X(sen e cos) para Y(sen e cos))
         retorno: tupla (coeficientes, número de coeficientes)
    ''' 
    coeffs = elliptic_fourier_descriptors(contour, order= n_coeffs, normalize=True)
    return (coeffs.flatten()[3:(n_coeffs*4+1)], (n_coeffs*4+1 - 3))

# NÃO ESTÁ FUNCIONANDO! Calcula Dimensão Fractal  - 
# From: https://github.com/jankaWIS/fractal_dimension_analysis/blob/main/fractal_analysis_fxns.py
def fractal_dimension(Z, threshold=0.9):
    """
    calculate fractal dimension of an object in an array defined to be above certain threshold as a count of squares
    with both black and white pixels for a sequence of square sizes. The dimension is the a coefficient to a poly fit
    to log(count) vs log(size) as defined in the sources.
    :param Z: np.array, must be 2D
    :param threshold: float, a thr to distinguish background from foreground and pick up the shape, originally from
    (0, 1) for a scaled arr but can be any number, generates boolean array
    :return: coefficients to the poly fit, fractal dimension of a shape in the given arr
    """
    # Only for 2d image
    assert (len(Z.shape) == 2)

    # Transform Z into a binary array
    Z = (Z < threshold)

    # Minimal dimension of image
    p = max(Z.shape)

    # Greatest power of 2 less than or equal to p
    n = 2 ** np.floor(np.log(p) / np.log(2))

    # Extract the exponent
    n = int(np.log(n) / np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2 ** np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))

    # Fit the successive log(sizes) with log (counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

def boxcount(Z, k):
    """
    returns a count of squares of size kxk in which there are both colours (black and white), ie. the sum of numbers
    in those squares is not 0 or k^2
    Z: np.array, matrix to be checked, needs to be 2D
    k: int, size of a square
    """
    S = np.add.reduceat(
        np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
        np.arange(0, Z.shape[1], k), axis=1)  # jumps by powers of 2 squares

    # We count non-empty (0) and non-full boxes (k*k)
    #return len(np.where((S > 0) & (S < k * k))[0])
    return len(np.where(S > 0)[0])

#Calcula as distâncias radiais (algumas métricas de \cite{Po-HsiangTsui2010a, e \cite{Chiang2001} para CA de mama})
def radial_distances_stats(centroid, border_coords):
    ''' Retorna:
        desvio padrão da distancia radial (SDNRL)
        tx de área: porcentagem de area fora da distância radial média
        RI: indice de rugosidade
        E : entropia do histograma do comprimento radial normalizado representa a redondeza e a rugosidade.
        k: kurtosis do histograma das 
        MRD: comprimento radial máximo (maximum length from center of gravity to perimeter) - valores absolutos (não normalizados)
        ARD: comprimento radial médio (average length from center of gravity to perimeter) - valores absolutos (não normalizados)
    '''    
    dis= cdist(border_coords, list([centroid]), metric='euclidean') 
    dis_Norm = dis/np.max(dis)   #valores noramalizados entre 0 e 1
    mean = np.mean(dis_Norm)     #valores noramalizados entre 0 e 1
    SDNRL= np.std(dis_Norm, ddof=1)  #valores noramalizados entre 0 e 1
    
    # Calcula taxa de area, soma di - dis[i+1]:
    N = dis_Norm.shape[0]
    area_out = 0
    sum = 0
    for i in range(N):
        if dis_Norm[i] >=mean: 
           area_out +=(dis_Norm[i] - mean)   
        if i != (N-1):
            sum += np.abs(dis_Norm[i] - dis_Norm[i+1])
    RA = area_out /(N*mean)
    
    # Calcula índice de rugosidade(RI):
    RI = sum/N
    
    # Calcula a entropia (E) do histograma dis_Norm
    hist, _ = np.histogram(dis, bins=100, density=True, )
    E = 0
    for p in hist: 
       if p!= 0:
          E+=(p*np.log(p))
    K = kurtosis(dis)  
    #print(K.shape, K)

    return ({'SDNRL': SDNRL, 'RA': RA[0], 'RI': RI[0], 'E':-E, 'K': K[0], 'MRD': np.max(dis), 'ARD': np.mean(dis)})
        
def get_list_features(xls_file):
   df = pd.pandas.read_excel(xls_file)
   feature_labels = []
   s = df.keys()
   aux = [ 'area_NC', 'perimetro_NC', 'major_axis_NC', 'minor_axis_NC', 'nucleus_position', \
           'sub_major_axis_angle_NC', 'convexity_NC']
   for i in np.arange(s.size):
       if df[s[i]].values[0] == 's':
            if s[i]  in aux:
                feature_labels.append(s[i])
            else:         
                feature_labels.append(s[i]+'N')
                feature_labels.append(s[i]+'C')
   return feature_labels  

def create_dictionary_masks_segmentation():
   # n_efd_coeffs: número de coefficientes a considerar (série Eliptica de fourier - EFD) para N e C
   feature_labels=['image_id', 'cell_id', 'bethesda', 'image_filename',
                    'nparray_points_segm_Nucleus', 'nparray_points_segm_Cyto']
   aux = [[] for i in range(len(feature_labels))]
   return dict(zip(feature_labels, aux))

## Gera dataframe com dados de mascaras cada celula:
def make_masks_DF(df_nucleos, df_cyto, df_nucleos_full, df_cyto_full):
  
   img = CRIC_images()
   count_cells = np.zeros(6, dtype = int)
   
   data_masks = create_dictionary_masks_segmentation() 
   ## delete pos test:
   cont = 0
   for image_id, cell_id, image_filename in df_nucleos.values:   
        cell = f'{image_id:05d}_{cell_id:05d}_'
        #cont = cont + 1
        ## delete pos test:
        #if cont > 20:
        #    break
       
        points_nucleos = df_nucleos_full.query('image_id == '+str(image_id) + ' and cell_id == '+str(cell_id))[['x', 'y']].values
        points_cyto = df_cyto_full.query('image_id == '+str(image_id) + ' and cell_id == '+str(cell_id))[['x', 'y']].values
        bethesda = img.Bethesda_classes[df_nucleos_full.query('image_id == '+str(image_id) + ' and cell_id == '+str(cell_id))['bethesda_system'].values[0]]
             
        # Eliminate duplicate e find complete contour points:  
        points_N = np.array(points_nucleos)
        points_N = np.unique(points_N, axis=0)
        rrN, ccN = polygon_perimeter(points_nucleos[:,1], points_nucleos[:,0])  # no arquivo (coluna, linha) aqui (linha, coluna)
        border_coords_N = [[ri, ci] for ri,ci in zip(rrN, ccN)] 
        points_N = np.array(border_coords_N)

        points_C = np.array(points_cyto)
        points_C = np.unique(points_C, axis=0)
        rrC, ccC = polygon_perimeter(points_cyto[:,1], points_cyto[:,0])
        border_coords_C = [[ri, ci] for ri,ci in zip(rrC, ccC)]
        points_C = np.array(border_coords_C)
        
        # Registry in dict:
        data_masks['image_id'].append(image_id)
        data_masks['cell_id'].append(cell_id)
        data_masks['image_filename'].append(image_filename)
        data_masks['bethesda'].append(bethesda)
        data_masks['nparray_points_segm_Nucleus'].append(points_N)
        data_masks['nparray_points_segm_Cyto'].append(points_C)

        count_cells[bethesda]+=1

   df = pd.DataFrame.from_dict(data_masks)   
   return (count_cells, df)  


def show_points(df_cell_masks, image_id, cell_id, escala=10):
        img = CRIC_images()
        # 'show' values: NU - Nucleus, CY - Cytoplasm, CL          
        
        points_N = df_cell_masks.query('image_id == '+str(image_id) + ' and cell_id == '+str(cell_id))['nparray_points_segm_Nucleus'][0]
        points_C = df_cell_masks.query('image_id == '+str(image_id) + ' and cell_id == '+str(cell_id))['nparray_points_segm_Cyto'][0]
         
        #points_N = points_N.astype(np.array)
        #print(points_N)
        
        # Calc regionprops metrics from Nucleos mask:
        mask_nucleo =  np.zeros((img.IMG_H, img.IMG_W), dtype=np.uint8)
        rr, cc = polygon(points_N[:,0], points_N[:,1])
        mask_nucleo[rr,cc] = 1
        plt.imshow(mask_nucleo, cmap = 'gray') 
        m_N = measure.regionprops(mask_nucleo)
        centroid_N = m_N[0].centroid   #centroid da mascara não do perimetro
       
        rrN = (points_N[:,0] - centroid_N[0]).astype(np.int32)
        ccN = (points_N[:,1] - centroid_N[1]).astype(np.int32)  
        rrN = rrN + np.abs(np.min(rrN))
        ccN = ccN + np.abs(np.min(ccN))
    
        ## Escalar imagem: 
        margem_rr = int(np.floor(np.max(rrN)*0.3))
        margem_cc = int(np.floor(np.max(ccN)*0.3))
        img = np.zeros((np.max(rrN)+2*margem_rr, np.max(ccN)+2*margem_cc), dtype=np.uint8)
        print(np.max(rrN), np.max(ccN), margem_rr, margem_cc)
        img[rrN+margem_rr, ccN+margem_cc] = 1
        res = cv2.resize(img, dsize=(img.shape[0]*escala, img.shape[1]*escala), interpolation=cv2.INTER_CUBIC)

        #plt.imshow(res, cmap = 'gray') 
        ind = np.where(img==1)     
     
        fig, axs = plt.subplots(1,2)
        fig.suptitle('centralized scaled mask (CSM)')
        axs[0].imshow(img, cmap = 'gray')
          
        #fig, axs = plt.subplots(1)
        fig.suptitle('Scatter  plot of CSM')
        axs[1].scatter(ind[1], ind[0]) #, s=0.5, marker='.')
        #axs[2].imshow(mask_nucleo, cmap = 'gray')
        #fig.tight_layout()


def create_dictionary_features():
    list_all = ["bethesda", "image_id", "cell_id",'areaN', 'perimeterN',  'major_axisN', 'minor_axisN', \
                'equivalent_diameterN', 'eccentricityN',  'circularityN', 'convexityN', 'solidityN', \
                'extentN', 'radial_distance_maxN', 'radial_distance_meanN', 'radial_distance_sdN', \
              'RAN', 'RIN', 'radial_distance_EN', 'radial_distance_kurtoseN', 'FDN', \
              'Use_curv1N', 'Use_curv2N', 'Use_curv3N', 'major_axis_angleN', \
             'areaC', 'perimeterC',  'major_axisC', 'minor_axisC', \
             'equivalent_diameterC', 'eccentricityC',  'circularityC', 'convexityC', 'solidityC', \
             'extentC', 'radial_distance_maxC', 'radial_distance_meanC', 'radial_distance_sdC', \
              'RAC', 'RIC', 'radial_distance_EC', 'radial_distance_kurtoseC', 'FDC', \
              'Use_curv1C', 'Use_curv2C', 'Use_curv3C', 'major_axis_angleC', \
             'area_NC', 'perimetro_NC', 'major_axis_NC', 'minor_axis_NC', 'nucleus_position', \
              'sub_major_axis_angle_NC', 'convexity_NC']  
    L = []
    for i in np.arange(len(list_all)):
        L.append([])
    return dict(zip(list_all, L))  

## Gera dataframe de FEATURES (por célula):
# Para cada célula (identificação, classe BETHESDA e features)
def make_stats(df_cell_masks):
   """ 
     Features (N e C):  
           area, eccentricity (da elipse), extent (area/boundingbox_area), 
           perímetro, major Axis, minor Axis, orientation, 
           circularidade (inversam ~ compacidade),
           convexidade (convexhul_perimeter/perimeter), 
           solidity (area/convex_hull_area),
           equivalent_diameter_area,
           FD (dimensão fractal), 
           ** Area_circulos_maiores_curvaturas/area_total          
           SDNRL, RA, RI, entropia das distâncias radiais (RD),
           kurtosis das distâncias radiais, maior RD, average RD
           ATUALIZAR!
       
     Features com relação entre N/C: 
           razão area N/C, razão perimetro N/C, razão convexidade N/C,
           soma_slopes_N_C, razão major axes N/C, razão minor axes N/C
           posição relativa do nucleo (em relação ao Citoplasma)  
   """ 
   data = create_dictionary_features()
   img = CRIC_images()
   #cont = 0
   count_cells = np.zeros(6, dtype = int)
   aux = df_cell_masks.keys()[0:3]
   for image_id, cell_id, bethesda in df_cell_masks[aux].values:   
        cell = f'{image_id:05d}_{cell_id:05d}_'
                
        ## contornos                                                             
        points_N = df_cell_masks.query('image_id == '+str(image_id) + ' and cell_id == '+str(cell_id))['nparray_points_segm_Nucleus'].values[0] 
        points_C = df_cell_masks.query('image_id == '+str(image_id) + ' and cell_id == '+str(cell_id))['nparray_points_segm_Cyto'].values[0]
                    
        # Calc regionprops metrics from Nucleos mask:
        mask_nucleo =  np.zeros((img.IMG_H, img.IMG_W), dtype=np.uint8)
        mask_nucleo[points_N[:,0],points_N[:,1]] = 1
        m_N = measure.regionprops(mask_nucleo)  
        # Calc regionprops metrics from Cyto mask:
        mask_cyto =  np.zeros((img.IMG_H, img.IMG_W), dtype=np.uint8)
        mask_cyto[points_C[:,0],points_C[:,1]] = 1
        m_C = measure.regionprops(mask_cyto)
 
         # Registry metrics on data dict:
        data['image_id'].append(image_id)
        data['cell_id'].append(cell_id)
        data['bethesda'].append(bethesda)

        # Get metrics for Nuclei:
        centN = m_N[0].centroid
        rdN= radial_distances_stats(centN, points_N)
        aN = m_N[0].area
        data['areaN'].append(aN)
        pN = m_N[0].perimeter
        data['perimeterN'].append(pN)
        maN = m_N[0].axis_major_length
        data['major_axisN'].append(maN)
        miN = m_N[0].axis_minor_length
        data['minor_axisN'].append(miN)       
        data['equivalent_diameterN'].append(m_N[0].equivalent_diameter_area)
        data['eccentricityN'].append(m_N[0].eccentricity)
        data['circularityN'].append((4*np.pi*aN)/np.power(pN,2))
        cN = measure.perimeter(m_N[0].image_convex)/pN
        data['convexityN'].append(cN)
        data['solidityN'].append(m_N[0].solidity)
        data['extentN'].append(m_N[0].extent)
        data['radial_distance_maxN'].append(rdN['MRD'])
        data['radial_distance_meanN'].append(rdN['ARD'])
        data['radial_distance_sdN'].append(rdN['SDNRL'])
        data['RAN'].append(rdN['RA'])
        data['RIN'].append(rdN['RI'])
        data['radial_distance_EN'].append(rdN['E'])
        data['radial_distance_kurtoseN'].append(float(rdN['K']))
        fdn = fractal_dimension(mask_nucleo)
        data['FDN'].append(fractal_dimension(mask_nucleo))
        m_curvs, zero_cross, max_points_mcurv, maximos = calc_curvatures(points_N[:,0],points_N[:,1])
        circle_area_ration1 = ((1/maximos)**2*np.pi)/aN
        data['Use_curv1N'].append(circle_area_ration1[0])
        data['Use_curv2N'].append(circle_area_ration1[1])
        data['Use_curv3N'].append(circle_area_ration1[2])
        angleN = m_N[0].orientation
        data['major_axis_angleN'].append(angleN)
    
        # Get metrics for Cyto:
        centC = m_C[0].centroid
        rdC= radial_distances_stats(centC, points_C)
        aC = m_C[0].area
        data['areaC'].append(aC)
        pC = m_C[0].perimeter
        data['perimeterC'].append(pC)
        maC = m_C[0].axis_major_length
        data['major_axisC'].append(maC)
        miC = m_C[0].axis_minor_length
        data['minor_axisC'].append(miC)       
        data['equivalent_diameterC'].append(m_C[0].equivalent_diameter_area)
        data['eccentricityC'].append(m_C[0].eccentricity)
        data['circularityC'].append((4*np.pi*aC)/np.power(pC,2))
        cC = measure.perimeter(m_C[0].image_convex)/pC
        data['convexityC'].append(cC)
        data['solidityC'].append(m_C[0].solidity)
        data['extentC'].append(m_C[0].extent)
        data['radial_distance_maxC'].append(rdC['MRD'])
        data['radial_distance_meanC'].append(rdC['ARD'])
        data['radial_distance_sdC'].append(rdC['SDNRL'])
        data['RAC'].append(rdC['RA'])
        data['RIC'].append(rdC['RI'])
        data['radial_distance_EC'].append(rdC['E'])
        data['radial_distance_kurtoseC'].append(rdC['K'])
        fdc = fractal_dimension(mask_cyto)
        data['FDC'].append(fractal_dimension(mask_cyto))
        m_curvs, zero_cross, max_points_mcurv, maximos = calc_curvatures(points_C[:,0],points_C[:,1])
        circle_area_ration2 = ((1/maximos)**2*np.pi)/aC
        data['Use_curv1C'].append(circle_area_ration2[0])
        data['Use_curv2C'].append(circle_area_ration2[1])
        data['Use_curv3C'].append(circle_area_ration2[2])
  
        angleC = m_C[0].orientation
        data['major_axis_angleC'].append(angleC)
        
        ## features involving N and C
        data['area_NC'].append(aN/aC)
        data['perimetro_NC'].append(pN/pC)
        data['major_axis_NC'].append(maN/maC)
        data['minor_axis_NC'].append(miN/miC)
        data['nucleus_position'].append(nucleus_position(centN, centC, maC))
        data['sub_major_axis_angle_NC'].append(angleN - angleC)
        data['convexity_NC'].append(cN/cC)	
        count_cells[bethesda]+=1
  
   df = pd.DataFrame(data)
   return (count_cells, df)  

#-----
 #Funções para normalizar (todos os dados):
## Normaliza dados
def normalize(min, max, val):
    n = (val - min)
    d = (max - min) 
    print(max, min)
    try:
        q = (n)/d
    except RuntimeWarning:
        print(" Erro ...")
        print(max, min)
    return (q)

def normalize_prop(prop, df):
    min = np.min(df[prop].values)
    max = np.max(df[prop].values)
    print(prop)
    return (normalize(min, max, df[prop].values))

# Filtra/Normaliza dados
def normalize_dataset(df):
  dataset = df.copy()
  dataset.areaN = normalize_prop('areaN', df)
  dataset.perimeterN = normalize_prop('perimeterN', df)
  dataset.major_axisN = normalize_prop('major_axisN', df)
  dataset.minor_axisN = normalize_prop('minor_axisN', df)
  dataset.equivalent_diameterN = normalize_prop('equivalent_diameterN',df)
  dataset.eccentricityN = normalize_prop('eccentricityN', df)
  dataset.circularityN  = normalize_prop('circularityN', df)
  dataset.convexityN = normalize_prop('convexityN', df)
  dataset.solidityN = normalize_prop('solidityN', df)   
  dataset.extentN = normalize_prop('extentN', df)      
  dataset.radial_distance_maxN = normalize_prop('radial_distance_maxN', df)  
  dataset.radial_distance_meanN = normalize_prop('radial_distance_meanN', df)
  dataset.radial_distance_sdN = normalize_prop('radial_distance_sdN', df)
  dataset.RAN = normalize_prop('RAN', df)
  dataset.RIN = normalize_prop('RIN', df) 
  dataset.radial_distance_EN = normalize_prop('radial_distance_EN', df)
  dataset.radial_distance_kurtoseN = normalize_prop('radial_distance_kurtoseN', df) 
  dataset.FDN = normalize_prop('FDN', df) 
  dataset.Use_curv1N = normalize_prop('Use_curv1N', df)
  dataset.Use_curv2N = normalize_prop('Use_curv2N', df) 
  dataset.Use_curv3N = normalize_prop('Use_curv3N', df) 
  dataset.major_axis_angleN = normalize_prop('major_axis_angleN', df)

  dataset.areaC = normalize_prop('areaC', df)
  dataset.perimeterC = normalize_prop('perimeterC', df)
  dataset.major_axisC = normalize_prop('major_axisC', df)
  dataset.minor_axisC = normalize_prop('minor_axisC', df)
  dataset.equivalent_diameterC = normalize_prop('equivalent_diameterC',df)
  dataset.eccentricityC = normalize_prop('eccentricityC', df)
  dataset.circularityC  = normalize_prop('circularityC', df)
  dataset.convexityC = normalize_prop('convexityC', df)
  dataset.solidityC = normalize_prop('solidityC', df)   
  dataset.extentC = normalize_prop('extentC', df)      
  dataset.radial_distance_maxC = normalize_prop('radial_distance_maxC', df)  
  dataset.radial_distance_meanC = normalize_prop('radial_distance_meanC', df)
  dataset.radial_distance_sdC = normalize_prop('radial_distance_sdC', df)
  dataset.RAC = normalize_prop('RAC', df)
  dataset.RIC = normalize_prop('RIC', df) 
  dataset.radial_distance_EC = normalize_prop('radial_distance_EC', df)
  dataset.radial_distance_kurtoseC = normalize_prop('radial_distance_kurtoseC', df) 
  dataset.FDC = normalize_prop('FDC', df) 
  dataset.Use_curv1C = normalize_prop('Use_curv1C', df)
  dataset.Use_curv2C = normalize_prop('Use_curv2C', df) 
  dataset.Use_curv3C = normalize_prop('Use_curv3C', df) 
  dataset.major_axis_angleC = normalize_prop('major_axis_angleC', df)

  dataset.area_NC = normalize_prop('area_NC', df)
  dataset.perimetro_NC = normalize_prop('perimetro_NC', df)
  dataset.major_axis_NC = normalize_prop('major_axis_NC', df)
  dataset.minor_axis_NC = normalize_prop('minor_axis_NC', df)
  dataset.nucleus_position = normalize_prop('nucleus_position', df)
  dataset.sub_major_axis_angle_NC = normalize_prop('sub_major_axis_angle_NC', df)
  dataset.convexity_NC = normalize_prop('convexity_NC', df)
  
  return dataset
