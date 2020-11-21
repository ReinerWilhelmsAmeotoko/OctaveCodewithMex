## Copyright (C) 2020 reiner
## 
## This program is free software; you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*- 
## @deftypefn {} {@var{[Pcounts, Ext, Psi, phi]} =} SchroedingerPlane(@var{nrounds}, @var{kBT=2}, @var{zeta=0.25}, @var{deltaTime=0},@var{doSimulation=false} )
## @end deftypefn

## Author: reiner <reiner@reiner-Z97P-D3>
## Created: 2020-11-08

function SchroedingerTwoSlit(nrounds, deltaTime=0.001)
  useVariableEnergyLandscape=false;
  twoslit = true;
  makemovie=true;
  picturecount=0;  % trick out ffmpeg's stupid counting method.
  forceboundary=false;
  modnumber = 1;   % plot figure 1 every modnumber. choose 1 for everytime 
  nrepeat = 500;
  normthreshold=1.0e-10;
  initialPlots=false;    
  ngrid = 256;       
  Nx = ngrid;
  Ny = ngrid;
  hbar = 1;
  mass = 0.2;
  [xp, yp] = meshgrid([0:1:Nx-1],[0:1:Ny-1]);
  xc = xp+0.5;
  yc = yp+0.5;    % centroids (xc,yc)
  dx = 1; dy=1;   % used in finite differences.
  noffs=0;
  scal = 1;
  
  %% ---------   -----+-----   
  wallheight=10;
  ncenter = 128;
  ncent = 16;
  nwide = 16;
  leftend = ngrid-ncent;
  nthick = 8;
  nystart = 124;
  
  plane = zeros(ngrid,ngrid);
  maskplane = ones(ngrid,ngrid);
  
  if (twoslit)
     plane = fillplane(plane,wallheight, nystart,  1,    nystart+nthick-1,  ncenter-ncent-nwide);
     plane = fillplane(plane,wallheight, nystart,  ncenter-ncent,  nystart+nthick-1, ncenter+ncent);
     plane = fillplane(plane,wallheight, nystart,  ncenter+ncent+nwide,  nystart+nthick-1, ngrid);
     maskplane = fillplane(maskplane,0, nystart,  1,    nystart+nthick-1,  ncenter-ncent-nwide);
     maskplane = fillplane(maskplane,0, nystart,  ncenter-ncent,  nystart+nthick-1, ncenter+ncent);
     maskplane = fillplane(maskplane,0, nystart,  ncenter+ncent+nwide,  nystart+nthick-1, ngrid);
  else
     wallheight=1;
     plane = fillplane(plane,wallheight, 1,1, ngrid, 3);
     plane = fillplane(plane,wallheight, 1, ngrid-2, ngrid, ngrid);
     plane = fillplane(plane,wallheight, 1, 1, 3, ngrid);
     plane = fillplane(plane,wallheight, ngrid-3, 1, ngrid, ngrid);
  endif;
  %Psi = zeros(ngrid,ngrid);
  %Psi = fillplane(Psi, 1, nlowy,nlowx,nhigy,nhix);
  %%Psi = Psi.*exp(i*0.25*yc);
  Psi = exp(-0.0005*((xc-128).^2+5*(yc-32).^2)).*(exp(i*0.97123*yc)); %% ...
  %%+ exp(-0.005*((xc-40).^2+3*(yc-220).^2)).*(0.01+exp(-i*0.1788*yc + i*0.134*xc));
  
  if (deltaTime == 0)
   deltaTime = 0.01;
  endif;

  figure(1); 
  colormap(jet(256));
  
  snorm = sum(sum(Psi.*conj(Psi)));
  Psi = Psi/sqrt(snorm); 
 
  for kk = 0:nrounds;
    viewpoint = [0 90]; 
    %viewpoint = [124.384    47.34];
    %%Psi = updateSchroedinger(Psi, plane, ngrid, deltaTime, hbar, mass);
    Psi = mexSchroedingerRKmulti(Psi,plane,nrepeat,deltaTime, hbar,mass);

    if (forceboundary)
      Psi = fillplane(Psi,0, nystart,  1,    nystart+nthick-1,  ncenter-ncent-nwide);
      Psi = fillplane(Psi,0, nystart,  ncenter-ncent,  nystart+nthick-1, ncenter+ncent);
      Psi = fillplane(Psi,0, nystart,  ncenter+ncent+nwide,  nystart+nthick-1, ngrid);      
    endif;
    snorm = sqrt(sum(sum(Psi.*conj(Psi))));
    if (snorm>1.005 || snorm < 0.995)
      printf("Houston we have a problem at step %i snorm=%10.6f\n",kk,snorm);
      break;
    endif 
    if (abs(snorm-1.0)>normthreshold)
       Psi = Psi/snorm; 
       printf("Normalization step %i snorm= %18.14f\n",kk,snorm);
       fflush(stdout);
    endif;   
    
    if (mod(kk,modnumber)==0)
      makeLittlePlot(xp,yp,ngrid, Nx,Ny, plane,maskplane, Psi, kk*nrepeat*deltaTime,viewpoint);
      %%makeSuperPlot(xp,yp,ngrid, Nx,Ny, plane, Psi,kk*nrepeat*deltaTime,viewpoint);
      drawnow;
      if (makemovie)
        picturecount = picturecount+1;
        filename=["movie/schroedinger_",num2str(picturecount,"%5.5i"),".png"];
        print(filename,'-dpng'); 
      endif;
     endif  
   endfor;
endfunction

function plane = fillplane(plane,value, lowy, lowx, highy, highx)  
  jj=[lowx:1:highx];   
  for k=lowy:highy,
    plane(k,jj)=value;
  endfor;
endfunction;

function Psi = updateSchroedinger(Psi, En, ngrid,dt, hbar, mass)
   %L = computeLaplacian(Psi,ngrid);
   L = mexLaplacian(Psi);
   coef = -hbar^2/(2*mass);
   coef2 = 1/(i*hbar);
   dH = coef2*(coef*L + En.*Psi);
   Psi = Psi + dt * dH;
   %snorm = sum(sum(sqrt(Psi.*conj(Psi))))/ngrid;
   %Psi = Psi/snorm;
endfunction
 
function makeLittlePlot(xp,yp,ngrid, Nx,Ny, Ext, maskplane, Psi,time,viewpoint)
      dx = 1; 
      dy = 1;
      widx=0.48;
      widy=0.48;
     
      titlefontsize=14;
      txtfsz=14;
      
      Apsi = abs(Psi);
      phase = arg(Psi).*maskplane;
      titletext = maketitleString(Apsi, "|\\Psi(x,t)|",time);
      
      % Apsi = Apsi.*maskplane;   % make walls visible.
  
      Apsi(1,1) = 0.02;
      surf(xp,yp,Apsi,'edgecolor','none');
      xlabel('x','fontsize',txtfsz); ylabel('y','fontsize',txtfsz); zlabel('{|\Psi|}','fontsize',txtfsz);
      title(titletext,'fontsize',titlefontsize);
      axis([0 Nx 0 Nx 0 1]);
      axis('equal');
      view(viewpoint);
      set(gca,'linewidth',2, 'fontsize',16);
      axis('off');
endfunction 
 
function makeSuperPlot(xp,yp,ngrid, Nx,Ny, Ext, Psi,time,viewpoint)
      dx = 1; 
      dy = 1;
      widx=0.48;
      widy=0.48;
      pos1 = [0.01  0.5  widx widy];
      pos2 = [0.51  0.5  widx widy];
      pos3 = [0.01  0.01 widx widy];      
      pos4 = [0.51  0.01 widx widy];

      figure(1); 
      clf ;
      set(1,'position',[40 20.000  1180 940]);
  
      titlefontsize=14;
      txtfsz=14;
      colormap(jet(256));
      
      subplot(2,2,1,'align');
      %subplot(2,2,1,'position',pos1);
      titletext = maketitleString(Ext, "E(x,t)",time);
      Ext(1,1) = -3;
      Ext(Nx,Ny)= 3;
      surf(xp,yp,Ext,'edgecolor','none');
      xlabel('x','fontsize',txtfsz); ylabel('y','fontsize',txtfsz); zlabel('E(x,t)','fontsize',txtfsz);
      title(titletext,'fontsize',titlefontsize);
      view(viewpoint);
      set(gca,'linewidth',2, 'fontsize',16);
      axis('off');
      
      subplot(2,2,2,'align'); 
      Apsi = abs(Psi);
      titletext = maketitleString(Apsi, "|\\Psi(x,t)|",time);
      Apsi(1,1)=0;
      Apsi(Nx,Ny)=0.01;
      surf(xp,yp,Apsi,'edgecolor','none');
      xlabel('x','fontsize',txtfsz); ylabel('y','fontsize',txtfsz); zlabel('{|\Psi|}','fontsize',txtfsz);
      title(titletext,'fontsize',titlefontsize);
      view(viewpoint);
      set(gca,'linewidth',2, 'fontsize',16);
      axis('off');
      
      subplot(2,2,3,'align'); 
      ag = arg(Psi);
      titletext = maketitleString(ag, "arg(\\Psi(x,t))",time);
      ag(1,1)=-pi;
      ag(Nx,Ny)=pi;
      surf(xp,yp,ag,'edgecolor','none');
      xlabel('x','fontsize',txtfsz); ylabel('y','fontsize',txtfsz); zlabel('{arg\Psi|}','fontsize',txtfsz);
      title(titletext,'fontsize',titlefontsize);
      view(viewpoint);
      set(gca,'linewidth',2, 'fontsize',16);
      axis('off');
      
endfunction 

function [str] = maketitleString(F, TeXname, time)
  mi = min(min(F));
  ma = max(max(F));
  str = sprintf("%s(t=%5.2f) [%5.3f, %5.3f]",TeXname,time,mi,ma);
endfunction
  
function [vx, vy] = computeNabla(F,ngrid,dx,dy);
opx = [-1 8 0 -8 1]/(12*dx);
opy = [-1 8 0 -8 1]/(12*dy);
vx = conv2(F,  opx,  'same'); 
vy = conv2(F,  opy', 'same');  % note the transpose.
endfunction;

function [divV] = computeDiv(F,ngrid,dx,dy);
opx = [-1 8 0 -8 1]/(12*dx);
opy = [-1 8 0 -8 1]/(12*dy);
[vx, vy] = computeNabla(F,ngrid,dx,dy);
divV = conv2(vx, opx, 'same') + conv2(vy,opy','same');
endfunction;

function [lap] = computeLaplacian(F,ngrid);
  c1=0; c2=-1/30; c3=-1/60; c4=4/15; c5=13/15; c6=-21/5;
  K = [c1,  c2, c3, c2, c1;
       c2,  c4, c5, c4, c2;
       c3,  c5, c6, c5, c3;
       c2,  c4, c5, c4, c2;
       c1,  c2, c3, c2, c1];
  lap = conv2(F,K,'same');
endfunction;

%function [lap] = computeLaplacian(F,ngrid);
%  K = [0    0  -1/12    0    0; 
%       0    0    4/3    0    0 ; 
%    -1/12  4/3  -5    4/3  -1/12; 
%       0    0    4/3    0    0; 
%       0    0   -1/12   0    0];
%  lap = conv2(F,K,'same');
%endfunction;
% compute a simple schedule for the parameter tau 
% to interpolate between two energy distributions
function tau = gimmiSchedule(kkx)
  ninterval = 1000;
  timestep = 1/ninterval;
  nvaryintervals = 5; 
  kk = kkx;
  if (kkx<nvaryintervals*ninterval)
    kk = mod(kkx,5*ninterval);
  else
    kk = kkx;
  endif;
  
  if (0 <= kk && kk <= ninterval)
    tau = kk*timestep;
    return;
  elseif ( ninterval<kk && kk <= 2*ninterval)
    tau = 1;
    return;
  elseif (2*ninterval<kk && kk <= 3*ninterval)
    tau = 1-(kk-2*ninterval)*timestep;
    return;
  elseif (3*ninterval<kk && kk <= 4*ninterval)
    tau = 0;
    return;
  elseif (4*ninterval<kk && kk <= 4.5*ninterval)
    tau = (kk-4*ninterval)*timestep;
    return; 
  else 
    tau = 1/2;
    return;
  endif
  tau = 1;
endfunction 
