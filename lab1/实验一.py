import cv2
import numpy as np

# ## 取反色
def ColorInvert(src,dst):
    #flag为1表示读取彩色图片
    img = cv2.imread(src,1)
    img_d = 255 - img
    cv2.imwrite(dst,img_d)

## 提高对比度
def Sharpen(src,dst):
    # dst = img1×α + img2×β + γ,将两张照片线性组合
    img = cv2.imread(src)
    img2 = np.ones((img.shape[0],img.shape[1],img.shape[2])).astype("uint8")
    Contrastimg = cv2.addWeighted(img, 1.5, img2, 2, 0)  # 调整对比度
    cv2.imwrite(dst,Contrastimg)

## 图片旋转
def ImageReverse(src,dst,mode='LR'):
    img = cv2.imread(src)
    if mode=="LR":
        img = np.flip(img, 1)
    elif mode=="UD":
        img = np.flip(img,0)
    elif mode=="both":
        img = np.flip(np.flip(img,1),0)
    cv2.imwrite(mode+dst,img)

##限制像素
def WippeColor(src,dst,n):
    img = cv2.imread(src)
    img = np.clip(img,0,n)
    cv2.imwrite(dst,img)

## 记录坐标
def ClassifyPixelByColor(src):
    img = cv2.imread(src)
    save = [[]]*6
    IND = 0
    for m in range(img.shape[2]):
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i][j][m]>200:save[2*m].append(img[i][j][m])
                else: save[2*m+1].append(img[i][j][m])
                IND+=1
    return tuple(save)

def ImageCompress(src,dst,k,type="svd"):
    img = cv2.imread(src)
    assert k <= min(img.shape[0],img.shape[1])
    compress_img = [] #用来装RGB三个通道的数组
    if type=="svd":
        for i in range(3):
            U,sigmma,VT = np.linalg.svd(img[:,:,i])
            im = np.matmul(np.matmul(U[:, :k], np.diag(sigmma[:k])), VT[:k, :])
            compress_img.append(im.astype('uint8'))
    if type=='PCA':
        for i in range(3):
            # 每列作为一个向量，先0均值化，再求协方差矩阵的特征值和特征向量，取前N个主成分，再重构图像
            data = np.asarray(img[:,:,i], np.double)
            # data -= np.mean(data, axis=0)
            # 以列为变量计算方式，计算协方差矩阵
            data_cov = np.cov(data, rowvar=False)
            feat_values, feat_vectors = np.linalg.eig(data_cov)
            feat_index = np.argsort(np.sqrt(feat_values), axis=0)[::-1]
            V = feat_vectors[:, feat_index]
            major_data = np.dot(np.dot(data, V[:, :k]), V[:, :k].T)
            compress_img.append(np.asarray(major_data, np.uint8))
    if type=="FT":
        for i in range(3):
            Cat_ft = np.fft.fft2(img[:,:,i]) #变换
            Cat_sort = np.sort(np.abs(Cat_ft.reshape(-1)))  # 按大小排序
            keep = k/min(img.shape[0],img.shape[1]) # 计算保持度
            thresh = Cat_sort[int(np.floor((1 - keep) * len(Cat_sort)))]
            ind = np.abs(Cat_ft) > thresh
            Alow = np.fft.ifft2(Cat_ft * ind).real #逆变换并取实部
            compress_img.append(Alow)
    img1 = np.stack(compress_img,axis=2)
    cv2.imwrite(f"{type}{k}_size_"+dst,img)

def ImageResize(src,dst,alpha=1.0,type="线性插值"):
    img = cv2.imread(src)
    h,w,c = img.shape
    h_a, w_a = int(h * alpha), int(w * alpha)
    resize_img = np.zeros((h_a, w_a, c))
    if type=="线性插值":
        #线性插值法,有损空间对称性质
        for m in range(c):
            for i in range(h_a):
                for j in range(w_a):
                    resize_img[i,j,m] = img[round(h*i/h_a),round(w*j/w_a),m]
    if type=="双线性插值":
        for n in range(c):
            for h_y in range(h_a):
                for w_x in range(w_a):
                    src_x = (w_x + 0.5) * alpha - 0.5
                    src_y = (h_y + 0.5) * alpha - 0.5
                    # 计算源图上四个近邻点的位置
                    src_x_0 = int(np.floor(src_x))
                    src_y_0 = int(np.floor(src_y))
                    src_x_1 = min(src_x_0 + 1, w - 1)
                    src_y_1 = min(src_y_0 + 1, h - 1)
                    # 双线性插值
                    value0 = (src_x_1 - src_x) * img[src_y_0, src_x_0, n] + (src_x - src_x_0) * img[src_y_0, src_x_1, n]
                    value1 = (src_x_1 - src_x) * img[src_y_1, src_x_0, n] + (src_x - src_x_0) * img[src_y_1, src_x_1, n]
                    resize_img[h_y, w_x, n] = int((src_y_1 - src_y) * value0 + (src_y - src_y_0) * value1)

if __name__ == "__main__":
    ImageCompress("dog.jpg","small_dog.jpg",100,type='FT')
    ImageResize("dog.jpg","resize_dog.jpg",0.8,type="线性插值")