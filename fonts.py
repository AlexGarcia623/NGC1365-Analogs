import matplotlib as mpl
mpl.use('agg')
import numpy as np
import  matplotlib.font_manager
import os
import matplotlib.pyplot as plt
flist = matplotlib.font_manager.get_fontconfig_fonts()
names = [matplotlib.font_manager.FontProperties(fname=fname).get_name() for fname in flist]

print(np.unique(names, return_counts=True))

os.environ['MANPATH'] = "/home/paul.torrey/local/texlive/2018/texmf-dist/doc/man:$MANPATH"
os.environ['INFOPATH']= "/home/paul.torrey/local/texlive/2018/texmf-dist/doc/info:$INFOPATH"
os.environ['PATH']    = "/home/paul.torrey/local/texlive/2018/bin/x86_64-linux:/home/paul.torrey/local/texlive/2018/texmf-dist:$PATH"

i=0
for FONT in names:
    plt.clf()
    print(FONT)
    plt.rcParams["font.family"] = FONT

    plt.plot([8.1,8.3],[8.1,8.3])

    plt.savefig('fonts/' + FONT + '.pdf')
    mpl.rcParams.update(matplotlib.rcParamsDefault)
    i+=1

c = [u'DejaVu Serif',
 u'URW Bookman',
 u'Nimbus Sans',
 u'Liberation Serif',
 u'Nimbus Mono PS', 
 u'Liberation Mono',
 u'Nimbus Sans Narrow',
 u'Nimbus Sans', 
 u'DejaVu Serif', 
 u'DejaVu Sans', 
 u'P052',
 u'Liberation Mono', 
 u'Nimbus Sans', 
 u'DejaVu Serif', 
 u'DejaVu LGC Sans Mono',
 u'Liberation Sans', 
 u'DejaVu Serif', 
 u'Nimbus Roman',
 u'DejaVu Sans', 
 u'Liberation Mono', 
 u'P052', 
 u'C059', 
 u'Liberation Sans', 
 u'DejaVu LGC Sans Mono', 
 u'Liberation Sans', 
 u'Nimbus Sans Narrow', 
 u'Nimbus Roman', 
 u'Liberation Sans', 
 u'DejaVu LGC Sans Mono', 
 u'DejaVu Serif', 
 u'URW Bookman', 
 u'Nimbus Roman', 
 u'Liberation Serif', 
 u'Nimbus Roman', 
 u'DejaVu Sans', 
 u'Nimbus Mono PS',
 u'D050000L', 
 u'Nimbus Sans Narrow', 
 u'P052', 
 u'DejaVu Sans', 
 u'DejaVu Sans', 
 u'DejaVu Serif', 
 u'C059', 
 u'DejaVu Sans', 
 u'DejaVu Sans', 
 u'URW Gothic', 
 u'URW Bookman', 
 u'URW Gothic',
 u'Liberation Serif', 
 u'DejaVu Sans Mono', 
 u'URW Gothic', 
 u'DejaVu LGC Sans Mono', 
 u'C059',
 u'Nimbus Mono PS',
 u'C059', 
 u'Nimbus Sans', 
 u'DejaVu Sans', 
 u'DejaVu Serif', 
 u'Nimbus Mono PS',
 u'URW Gothic',
 u'DejaVu Sans', 
 u'DejaVu Serif', 
 u'DejaVu Sans Mono', 
 u'Nimbus Sans Narrow', 
 u'URW Bookman', 
 u'DejaVu Sans Mono', 
 u'Z003',
 u'Liberation Serif', 
 u'DejaVu Sans Mono', 
 u'P052', 
 u'Liberation Mono']

print(c==names)