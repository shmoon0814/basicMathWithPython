import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math


def arrowed_spines(fig, ax, remove_ticks=False):
    """
    좌표축 화살표를 그리기 위한 함수
    https://stackoverflow.com/questions/33737736/matplotlib-axis-arrow-tip
    """
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # removing the default axis on all sides:
    for side in ['bottom','right','top','left']:
        ax.spines[side].set_visible(False)

    if remove_ticks == True:
        # removing the axis ticks
        plt.xticks([]) # labels
        plt.yticks([])
        ax.xaxis.set_ticks_position('none') # tick markers
        ax.yaxis.set_ticks_position('none')

    # get width and height of axes object to compute
    # matching arrowhead length and width
    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height

    # manual arrowhead width and length
    hw = 1./50.*(ymax-ymin)
    hl = 1./25.*(xmax-xmin)
    lw = 1. # axis line width
    ohg = 0.4 # arrow overhang

    # compute matching arrowhead length and width
    yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width
    yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height

    # draw x and y axis
    ax.arrow(xmin, 0, xmax-xmin, 0., fc='k', ec='k', lw = lw,
             head_width=hw, head_length=hl, #overhang = ohg,
             length_includes_head= True, clip_on = False)

    ax.arrow(0, ymin, 0., ymax-ymin, fc='k', ec='k', lw = lw,
             head_width=yhw, head_length=yhl, #overhang = ohg,
             length_includes_head= True, clip_on = False)


mpl.style.use('bmh')
mpl.style.use('seaborn-whitegrid')

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(1, 1, 1)
ax.xaxis.set_tick_params(labelsize=18)
ax.yaxis.set_tick_params(labelsize=18)
plt.xlabel('$x$', fontsize=25)
plt.ylabel('$y$', fontsize=25)

x = np.linspace(-3, 2, 10)
y = 2*x+4
ax.plot(x,y,'k')

arrowed_spines(fig, ax)

#plt.show()


# ---------- 밑이 2, 3, 4, 1/2, 1/3, 1/4 인 지수함수 그리기

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
fig.set_size_inches((15,6))

ax1.xaxis.set_tick_params(labelsize=18)
ax1.yaxis.set_tick_params(labelsize=18)

x = np.linspace(-2, 2, 100)
a1, a2, a3 = 2, 3, 4
y1, y2, y3 = a1**x, a2**x, a3**x

ax1.plot(x, y1, color='k', label=r"$2^x$")
ax1.plot(x, y2, '--', color='k', label=r"$3^x$")
ax1.plot(x, y3, ':', color='k', label=r"$4^x$")

a1, a2, a3 = 1/2, 1/3, 1/4
y1, y2, y3 = a1**x, a2**x, a3**x

ax2.plot(x, y1, color='k', label=r"$1/2^x$")
ax2.plot(x, y2, '--', color='k', label=r"$1/3^x$")
ax2.plot(x, y3, ':', color='k', label=r"$1/4^x$")

ax1.set_xlabel('$x$', fontsize=25)
ax1.set_ylabel('$y$', fontsize=25)
ax1.legend(fontsize=20)

ax2.set_xlabel('$x$', fontsize=25)
ax2.set_ylabel('$y$', fontsize=25)
ax2.legend(fontsize=20)

ax1.xaxis.set_tick_params(labelsize=18)
ax1.yaxis.set_tick_params(labelsize=18)

ax2.xaxis.set_tick_params(labelsize=18)
ax2.yaxis.set_tick_params(labelsize=18)

arrowed_spines(fig, ax1)
arrowed_spines(fig, ax2)


#plt.show()

# log 함수 밑 재정의
def log(x, base=np.e):
    return np.log(x) / np.log(base)


print(log(4, 2))
print(log(100, 10))

# 로그함수 그래프 그려보기
fig1, (ax3, ax4) = plt.subplots(1, 2, sharex=True, sharey=True)

x = np.linspace(0.01, 5, 100)
y1, y2 = np.log10(x), np.log(x)

ax3.plot(x, y1, color='k', label=r"$log10x")
ax3.plot(x, y2, '--', color='k', label=r"$logex")

ax3.set_xlabel('$x$', fontsize=25)
ax3.set_ylabel('$y$', fontsize=25)
ax3.legend(fontsize=20)

#divide by zero encountered in log 에러가 뜨는데 처리 범위를 넘어선다고 오류를 뱉는 것이다.  0, 5 범위가 맞을것인데.. -> 0.01로 변경하면 발생안함

y3, y4 = log(1/10, x), log(1/np.e, x)
ax4.plot(x, y3, color='k', label=r"$log10x")
ax4.plot(x, y4, '--', color='k', label=r"$logex")

ax4.set_xlabel('$x$', fontsize=25)
ax4.set_ylabel('$y$', fontsize=25)
ax4.legend(fontsize=20)

arrowed_spines(fig1, ax3)
arrowed_spines(fig1, ax4)

#plt.show()


#로그함수 깃허브 풀이
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
fig.set_size_inches((15, 6))

ax1.xaxis.set_tick_params(labelsize=18)
ax1.yaxis.set_tick_params(labelsize=18)

x1 = np.linspace(0.0001, 5, 1000) #0을 넣으면 안됨 항상 x 는 항상 양수여야 하기 때문이다..
x2 = np.linspace(0.01, 5, 100)
y1, y2 = log(x1, 10), log(x2, np.e)

ax1.plot(x1, y1, label=r"$\log_{10} x$", color='k')
ax1.plot(x2, y2, '--', label=r"$\log_{e} x$", color='k')

ax1.set_xlabel('$x$', fontsize=25)
ax1.set_ylabel('$y$', fontsize=25)
ax1.legend(fontsize=20, loc='lower right')

ax2.xaxis.set_tick_params(labelsize=18)
ax2.yaxis.set_tick_params(labelsize=18)

x1 = np.linspace(0.0001, 5, 1000)
x2 = np.linspace(0.01, 5, 100)
y1, y2 = log(x1, 1 / 10), log(x2, 1 / np.e)

ax2.plot(x1, y1, label=r"$\log_{1/10} x$", color='k')
ax2.plot(x2, y2, '--', label=r"$\log_{1/e} x$", color='k')

ax2.set_xlabel('$x$', fontsize=25)
ax2.set_ylabel('$y$', fontsize=25)
ax2.legend(fontsize=20, loc='upper right')

arrowed_spines(fig, ax1)
arrowed_spines(fig, ax2)

#plt.show()

#로지스틱 시그모이드 함수
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(1, 1, 1)

ax.xaxis.set_tick_params(labelsize=18)
ax.yaxis.set_tick_params(labelsize=18)

z = np.linspace(-6, 6, 100)
#sigma = 1 / (1 + np.exp(-z)) #np.exp 상수가 np.e ** z 이다
sigma = 1 / (1 + np.e ** -z) #np.exp 상수가 np.e ** z 이다

ax.plot(z, sigma, color='k')

ax.set_xlabel('$z$', fontsize=25)
ax.set_ylabel(r'$\sigma(z)$', fontsize=25)

arrowed_spines(fig, ax)

plt.show()