#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 14:30:55 2018

@author: louis
"""

import numpy as np
import pytz
from datetime import datetime,timedelta

class WH300_File:
    def __init__(self,fname=None,year=1970):

        infile=open(fname,'rb')
        tout=np.frombuffer(infile.read(),dtype='u1')
        infile.close()
        
        # Find start of frames.
        istart=np.where((tout[:-4]==0x7f)*(tout[1:-3]==0x7f)*(tout[4:]==0))[0]
        NbBytes=2+int(np.median(tout[istart+2]+256*tout[istart+3]))

        istart=istart[(0<=istart)*(istart+NbBytes<=len(tout))]
        
        data=np.array([tout[d:d+NbBytes] for d in istart])
        Chk =np.sum(data[:,:-2],axis=1)%65536
        ChkF=data[:,-1]*256+data[:,-2]
        data=data[Chk==ChkF,:]
        
        self.V={}
        self.V['BH']=WH300_File.ParseBH(data[:,:])
        
        NDT=self.V['BH']['NbData']

        for u in range(NDT):
            dts=self.V['BH']['DTOffsets'][u]
            dte=self.V['BH']['DTOffsets'][u+1] if u<NDT-1 else NbBytes+2
            t=WH300_File.ParseVariable(data[:,dts:dte],year=year)

            self.V[t[0]]=t[1]

        
        
        self.tout=tout
        self.data=data

    def ParseBH(data):
        VId=np.unique(data[:,0:2],axis=0)
        if VId.shape[0]>1:
            raise Exception('Variables not in the same order in all pings.')
            
        NDTypes=np.unique(data[:,5],axis=0)
        if len(NDTypes)>1:
            raise Exception('Not the same number of data types in all pings.')

        data=data[0,:]
        return dict(NbBytes  =data[2]+256*data[3],
                    NbData   =data[5],
                    DTOffsets=np.array([data[6+u*2]+256*data[6+u*2+1] for u in range(data[5])]))
        
    def ParseVariable(data,year):
        VId=[d[0]+d[1]*256 for d in np.unique(data[:,0:2],axis=0)]
        if len(VId)>1:
            raise Exception('Variables not in the same order in all pings.')
        Magic=VId[0]

        # Le champ de 'Variable Attitude' a de nombreux formats bizarres.
        # On renvoie tout sur un seul d'entre eux.
        Magic=int('3040',16) if ((int('3040',16)<=Magic)*(Magic<=int('30FC',16))) else Magic
        
        # Sur certains fichiers tres courts, on a des champs bizarres.
        Magic=1 if Magic in (1,129,257,513,769,1025,1281) else Magic
       
        VId={0:'FL',1:'Pad',128:'VL',
             256:'Vel',512:'Cor',768:'Amp',1024:'PGd',1280:'Status',
             1536:'BT',8192:'Nav',12288:'FA',int('3040',16):'VA'}[Magic]
        
        if VId=='FL':
            return [VId,WH300_File.ParseFL(data)]
        if VId=='VL':
            return [VId,WH300_File.ParseVL(data,year=year)]
        if VId=='Vel':
            return [VId,WH300_File.ParseVel(data)]
        if VId in ('Cor','Amp','PGd','Status'):
            return [VId,WH300_File.ParseByte(data)]
        if VId=='BT':
            return [VId,WH300_File.ParseBT(data)]
        if VId=='Nav':
            return [VId,WH300_File.ParseNav(data)]
        if VId=='FA':
            return [VId,WH300_File.ParseFA(data)]
        if VId=='VA':
            return [VId,WH300_File.ParseVA(data)]
        if VId=='Pad':
            return [VId,data]
    
    def ParseFL(data):
        data=np.unique(data,axis=0)[0]
        return dict(FWVer      =data[2]+data[3]/100,
                    SysConf    =data[4]+data[5]*256,
                    
                    Frequency  ={0:75,1:150,2:300,3:600,4:1200,5:2400,6:38}[data[4]&0x7],
                    Convex     =((data[4]&8)==8),
                    SnsConf    =(data[4]&48)>>4,
                    XdcrAtt    =((data[4]&64)==64),
                    Up         =((data[4]&128)==128),
                    BeamAngle  ={0:15,1:20,2:30,4:0}[data[5]&3],
                    BConf      =data[5]>>4,
                    
                    RealSim    =data[6],
                    NBeams     =data[8],
                    NCells     =data[9],
                    NPings     =data[10]+data[11]*256,
                    CellDepth  =(data[12]+data[13]*256)/100,
                    BlankDepth =(data[14]+data[15]*256)/100,
                    
                    SPMode     =data[16],
                    CorrThresh =data[17],
                    NCodeReps  =data[18],
                    PGdMin     =data[19],
                    EVMax      =(data[20]+data[21]*256)/100,
                    
                    Tpp        =60*data[22]+data[23]+data[24]/100,

                    CoordXform =data[25],
                    HeadingAlgn=(((data[26]+data[27]*256+32768)%65536)-32768)/100,
                    HeadingBias=(((data[28]+data[29]*256+32768)%65536)-32768)/100,
                    SensorSrc  =data[30],
                    SensorAv   =data[31],
                    Bin1Dstnc  =(data[32]+data[33]*256)/100,
                    XmtLength  =(data[34]+data[35]*256)/100,
                    RefLayStart=data[36],
                    RefLayStop =data[37],
                    FalseTgt   =data[38],
                    XmtLagDst  =(data[40]+data[41]*256)/100,
                    CPUSN      =data[42:50],
                    BandWidth  =data[50]+data[51]*256,
                    SysPow     =data[52])
    def ParseVL(data,year):
        y=np.uint16(data[:,57])*100+data[:,58]
        y[y==0]=year;
        return dict(ENum           = data[:,2]+256*data[:,3]+65536*data[:,11],
                    time           =[datetime(*d,tzinfo=pytz.utc).timestamp()
                                     for d in zip(y,
                                                  data[:, 5],
                                                  data[:, 6],
                                                  data[:, 7],
                                                  data[:, 8],
                                                  data[:, 9],
                                                  data[:,10]*10000)],
                    BIT            = data[:,12]+256*data[:,13],
                    SoundSpeed     = data[:,14]+256*data[:,15],
                    TransducerDepth=(data[:,16]+256*data[:,17])/10,
                    
                    Heading        =(((data[:,18]+data[:,19]*256+32768)%65536)-32768)/100,
                    Pitch          =(((data[:,20]+data[:,21]*256+32768)%65536)-32768)/100,
                    Roll           =(((data[:,22]+data[:,23]*256+32768)%65536)-32768)/100,
                    
                    Salinity       = data[:,24]+256*data[:,25],
                    Temperature    =(((data[:,26]+data[:,27]*256+32768)%65536)-32768)/100,
                    
                    MPTMin         = data[:,28],
                    MPTSec         = data[:,29]+data[:,30]/100,
                    HdgStd         = data[:,31],
                    PtchStd        = data[:,32]/10,
                    RollStd        = data[:,33]/10,
                    ADC            = data[:,34:42],
                    ESW            = data[:,42]+256*data[:,43]+256**2*data[:,44]+256**3*data[:,45],
                    P0             =(((data[:,48]+256*data[:,49]+256**2*data[:,50]+256**3*data[:,51]+2**31)%2**32)-2**31)*10,
                    PStd           =(data[:,52]+256*data[:,53]+256**2*data[:,54]+256**3*data[:,55])*10
                    )

    def ParseVel(data):
        d=(((data[:,2::2]+data[:,3::2]*256+32768)%65536)-32768)/1000
        Np=d.shape[0];NC=d.shape[1]//4
        d.shape=[Np,NC,4]
        #d[np.abs(d)>20]=np.NaN
        return d

    def ParseByte(data):
        d=data[:,2:]
        Np=d.shape[0];NC=d.shape[1]//4
        d.shape=[Np,NC,4]
        return d
    
    def ParseBT(data):
        VD=(((data[:,24:32:2]+256*data[:,25:32:2]+32768)%65536)-32768)/1000
        VD[np.abs(VD)>20]=np.NaN
        return dict(NumPings  =data[:,2]+256*data[:,3],
                    CorrMin   =data[:,6],
                    AmpMin    =data[:,7],
                    Mode    =data[:,9],
                    EVelMax   =(data[:,10]+256*data[:,11])/1000,
                    Range   =(data[:,16:24:2]+256*data[:,17:24:2]+65536*data[:,77:81])/100,
                    Vel     =VD,
                    Corr    =data[:,32:36],
                    Amp     =data[:,37:40],
                    MaxDepth=data[:,70]+data[:,71]*256,
                    RSSI    =data[:,72:76],
                    Gain    =data[:,76])
    def ParseNav(data):
        return dict(FirstUTC=np.array([datetime(*d,tzinfo=pytz.utc).timestamp() for d in zip(data[:,4]+256*data[:,5],data[:,3],data[:,2])])
                             +(data[:, 6]+256*data[:, 7]+256**2*data[:, 8]+256**3*data[:, 9])/10000,
                    PCOffset=(((data[:,10]+256*data[:,11]+256**2*data[:,12]+256**3*data[:,13]+2**31)%2**32)-2**31)/1000,
                    FirstLat=(((data[:,14]+256*data[:,15]+256**2*data[:,16]+256**3*data[:,17]+2**31)%2**32)-2**31)*180/2**31,
                    FirstLon=(((data[:,18]+256*data[:,19]+256**2*data[:,20]+256**3*data[:,21]+2**31)%2**32)-2**31)*180/2**31,
                    LastUTC =np.array([datetime(*d,tzinfo=pytz.utc).timestamp() for d in zip(data[:,4]+256*data[:,5],data[:,3],data[:,2])])
                             +(data[:, 22]+256*data[:,23]+256**2*data[:,24]+256**3*data[:,25])/10000,
                    LastLat =(((data[:,26]+256*data[:,27]+256**2*data[:,28]+256**3*data[:,29]+2**31)%2**32)-2**31)*180/2**31,
                    LastLon =(((data[:,30]+256*data[:,31]+256**2*data[:,32]+256**3*data[:,33]+2**31)%2**32)-2**31)*180/2**31,
                    SMG     =((data[:,40]+256*data[:,41]+2**15)%2**16-2**15)*0.001,
                    CMG     =((data[:,42]+256*data[:,43]+2**15)%2**16-2**15)*180/2**15,
                    Flags   =data[:,46]+256*data[:,47],
                    ENum    =data[:,50]+256*data[:,51]+256**2*data[:,52]+256**3*data[:,53],
                    EPCTime =np.array([datetime(*d,tzinfo=pytz.utc).timestamp() for d in zip(data[:,54]+256*data[:,55],data[:,57],data[:,56])])
                             +(data[:,58]+256*data[:,59]+256**2*data[:,60]+256**3*data[:,61])/100,
                    Pitch   =(((data[:,62]+256*data[:,63]+2**15)%2**16)-2**15)*180/2**15,
                    Roll    =(((data[:,64]+256*data[:,65]+2**15)%2**16)-2**15)*180/2**15,
                    Heading =(((data[:,66]+256*data[:,67]+2**15)%2**16)-2**15)*180/2**15)

    def ParseFA(data):
        return dict(EE=np.hstack([np.hstack((data[:,u:u+1]//16,data[:,u:u+1]%16)) for u in range(2,10)]),
                    EF=((data[:,10]+128)%256)-128,
                    EH=np.hstack([((((data[:,11]+data[:,12]*256+32768)%65536)-32768)/100)[:,None],data[:,13:14]]),
                    EI=(((data[:,14]+data[:,15]*256+32768)%65536)-32768)/100,
                    EJ=(((data[:,16]+data[:,17]*256+32768)%65536)-32768)/100,
                    EP=np.hstack([(((data[:,18]+data[:,19]*256+32768)%65536)-32768)[:,None]/100,
                                  (((data[:,20]+data[:,21]*256+32768)%65536)-32768)[:,None]/100,
                                  data[:,22][:,None]]),
                    EU=data[:,23],
                    EV=(((data[:,24]+data[:,25]*256+32768)%65536)-32768)/100,
                    EZ=data[:,26:34])

    def ParseVA(data):
        return ((data[:,2::2]+data[:,3::2]*256+32768)%65536-32768)/100.0     
                    
                    
    

