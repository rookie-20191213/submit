#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 06:01:51 2022

@author: nephilim
"""
import math
import numpy as np
from matplotlib import pyplot,cm,colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

class FM_Migration(object):
    def __init__(self,dx,dt,epsilon):
        self.dx=dx
        self.dt=dt
        self.epsilon=epsilon
    
    def _nextpow2(self,a):
        return np.ceil(np.log(a)/np.log(2)).astype(int)
        
    def _mwhalf(self,n,percent=10.):
        m=int(math.floor(percent*n/100.))
        h=np.hanning(2*m)
        return np.hstack([np.ones([n-m]),h[m:0:-1]])

    def _fktran(self,D,t,x,ntpad,nxpad,ishift=1):
        # nsamp=D.shape[0]
        ntr=D.shape[1]
        specfx,f=self._fftrl(D,t,ntpad)
        if ntr<nxpad:
            ntr=nxpad
        spec=np.fft.ifft(specfx.T,n=ntr,axis=0).T
        kxnyq=1./(2.*(x[1] - x[0]))
        dkx=2.*kxnyq/ntr
        kx=np.hstack([np.arange(0,kxnyq,dkx),np.arange(-kxnyq,0,dkx)])
        if ishift:
            tmp=zip(kx,spec)
            tmp.sort()
            kx=[i[0] for i in tmp]
            spec=[i[1] for i in tmp]
        return spec,f,kx

    def _ifktran(self,spec,f,kx):
        nf,nkx=spec.shape
        # nfpad=2**self._nextpow2(len(f))
        nkpad=2**self._nextpow2(len(kx))
        if kx[0]<0.0:
            ind=kx >= 0.0
            kx=np.hstack([kx[ind],kx[np.arange(ind[0])]])
            spec=np.hstack([spec[:,ind],spec[:,np.arange(ind[0])]])
        else:
            ind=False
        if nkx<nkpad:
            nkx=nkpad
        specfx=np.fft.fft(spec,nkx)
        D,t=self._ifftrl(specfx,f)
        dkx=kx[1] - kx[0]
        xmax=1.0/dkx
        dx=xmax/nkx
        x=np.arange(0,xmax,dx)
        return D,t,x

    def _fftrl(self,s,t,n):
        l=s.shape[0]
        m=s.shape[1]
        ntraces=1
        itr=0
        if l==1:
            nsamps=m
            itr=1
            s=s.T
        elif m==1:
            nsamps=l
        else:
            nsamps=l
            ntraces=m
        if nsamps!= len(t):
            t=t[0]+(t[1] - t[0])*np.arange(0,nsamps)
        if nsamps<n:
            s=np.vstack([s,np.zeros([n-nsamps,ntraces])])
            nsamps=n
        spec=np.fft.fft(s,n=nsamps,axis=0)
        spec=spec[:int(n/2)+1,:]
        fnyq=1./(2*(t[1] - t[0]))
        nf=spec.shape[0]
        df=2.0*fnyq/n
        f=df*np.arange(0,nf).T
        if itr:
            f=f.T
            spec=spec.T
        return spec,f

    def _ifftrl(self,spec,f):
        m,n=spec.shape
        itr=0
        if (m==1) or (n==1):
            if m==1:
                spec=spec.T
                itr=1
            nsamp=len(spec)
            # ntr=1
        else:
            nsamp=m
            # ntr=n
        nyq=0
        if (spec[-1]==np.real(spec[-1])).all():
            nyq=1
        if nyq:
            L1=np.arange(nsamp)
            L2=L1[-2:0:-1]
        else:
            L1=np.arange(nsamp)
            L2=L1[-2:0:-1]
        symspec=np.vstack([spec[L1,:],np.conj(spec[L2,:])])
        r=(np.fft.ifft(symspec.T)).real.T
        n=len(r)
        df=f[1] - f[0]
        dt=1.0/(n*df)
        t=dt*np.arange(n).T
        if itr==1:
            r=r.T
            t=t.T
        return r,t

    def fk_migration(self,profile):
        v=3e8/np.sqrt(self.epsilon)
        nsamp=profile.shape[0]
        ntr=profile.shape[1]
        t=np.arange(nsamp)*self.dt
        x=np.arange(ntr)*self.dx
        fnyq=1.0/(2.0*self.dt)
        # knyq=1.0/(2.0*self.dx)
        tmax=t[-1]
        xmax=np.abs(x[-1]-x[0])
        fmax=0.6*fnyq
        fwid=0.2*(fnyq - fmax)
        dipmax=85.0
        dipwid=90.0 - dipmax
        tpad=min([0.5*tmax,abs(tmax/math.cos(math.pi*dipmax/180.0))])
        xpad=min([0.5*xmax,xmax/math.sin(math.pi*dipmax/180.0)])
        nsampnew=int(2.0**self._nextpow2( round((tmax+tpad)/self.dt+1.0) ))
        tmaxnew=(nsampnew-1)*self.dt
        tnew=np.arange(t[0],tmaxnew+self.dt,self.dt)
        ntpad=nsampnew-nsamp
        profile=np.vstack([profile,np.zeros([ntpad,ntr])])
        ntrnew=2**self._nextpow2( round((xmax+xpad)/self.dx+1) )
        xmaxnew=(ntrnew-1)*self.dx+x[0]
        xnew=np.arange(x[0],xmaxnew+self.dx,self.dx)
        nxpad=ntrnew - ntr
        profile=np.hstack([profile,np.zeros([nsampnew,nxpad])])
        fkspec,f,kx=self._fktran(profile,tnew,xnew,nsampnew,ntrnew,0)
        df=f[1] - f[0]
        nf=len(f)
        ifmaxmig=int(round((fmax+fwid)/df+1.0))
        pct=100.0*(fwid/(fmax+fwid))
        fmask=np.hstack([self._mwhalf(ifmaxmig,pct),np.zeros([nf-ifmaxmig])])
        fmaxmig=(ifmaxmig-1)*df
        ve=v/2.0
        dkz=df/ve
        kz=(np.arange(0,len(f))*dkz).T
        kz2=kz**2
        th1=dipmax*math.pi/180.0
        th2=(dipmax+dipwid)*math.pi/180.0
        if th1==th2:
            print("No dip filtering")
        for j,kxi in enumerate(kx):
            fmin=np.abs(kxi)*ve
            ifmin=int(math.ceil(fmin/df))+1
            if th1!= th2:
                ifbeg=np.max([ifmin,1])+1
                ifuse=np.arange(ifbeg,ifmaxmig+1)
                if len(ifuse)<=1:
                    dipmask=np.zeros(f.shape)
                    dipmask[ifuse-1]=1
                else:
                    theta=np.arcsin(fmin/f[ifuse])
                    if1=int(np.round(fmin/(math.sin(th1)*df)))                
                    if1=np.max([if1,ifbeg])
                    if2=int(np.round(fmin/(math.sin(th2)*df)))
                    if2=np.max([if2,ifbeg])
                    dipmask=np.zeros(f.shape)
                    dipmask[if1:nf-1]=1
                    dipmask[if2:if1]=0.5+0.5*np.cos((theta[np.arange(if2,if1)-ifbeg]-th1)*math.pi/float(th2-th1))               
            else:
                dipmask=np.ones(f.shape)
            tmp=fkspec[:,j]*fmask*dipmask
            fmap=ve*np.sqrt(kx[j]**2+kz2)
            ind=np.vstack(np.nonzero(fmap<=fmaxmig)).T
            fkspec[:,j]*=0.0
            if len(ind)!=0:
                if fmap[ind].all()==0:
                    scl=np.ones(ind.shape[0])
                    li=ind.shape[0]
                    scl[1:li]=(ve*kz[ind[1:li]]/fmap[ind[1:li]])[:,0]
                else:
                    scl=ve*kz[ind]/fmap[ind]
                r_interp=scl.squeeze()*np.interp(fmap[ind],f,tmp.real).squeeze()
                j_interp=scl.squeeze()*np.interp(fmap[ind],f,tmp.imag).squeeze()
                fkspec[ind,j]=(r_interp+j_interp*1j).reshape([-1,1])
        Dmig,tmig,xmig=self._ifktran(fkspec,f,kx)
        Dmig=Dmig[:nsamp,:ntr]
        tmig=tmig[:nsamp]
        xmig=xmig[:ntr]
        return Dmig,tmig,xmig

if __name__=='__main__':
    # ep=np.load('OverThrust.npy')
    # ep_rms = np.sqrt(np.mean(ep**2))
    ep_rms=3.5
    profile=np.load('./without_ref_data.npy')
    FM_=FM_Migration(dx=8/profile.shape[1], dt=4e-11*500/profile.shape[0], epsilon=ep_rms)
    Dmig,tmig,xmig=FM_.fk_migration(profile)
    np.save('fk_migration.npy',Dmig)
    norm=colors.Normalize(vmin=-5e6, vmax=5e6)
    extent=(0,400,75,0)
    pyplot.figure(figsize=(8,6))
    pyplot.rcParams.update({'font.size': 15})

    gci=pyplot.imshow(Dmig,extent=extent,cmap=cm.gray_r,norm=norm)
    ax=pyplot.gca()
    ax.set_xticks(np.linspace(0,400,5))
    ax.set_xticklabels([0,2,4,6,8])
    ax.set_yticks(np.linspace(0,75,5))
    ax.set_yticklabels([0,5,10,15,20])
    
    pyplot.xlabel('Distance (m)')
    pyplot.ylabel('Time (ns)')
    divider=make_axes_locatable(ax)
    cax=divider.append_axes('right', size='4%', pad=0.15)
    cbar=pyplot.colorbar(gci,cax=cax)
    cbar.set_label('Amplitude')
    pyplot.savefig('00_Field_FKMigration', dpi=1000)
 