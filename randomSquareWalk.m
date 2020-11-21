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
## @deftypefn {} {@var{[Pcounts, Ext, Psi, phi]} =} randomSquareWalk (@var{nrounds}, @var{kBT=2}, @var{zeta=0.25}, @var{deltaTime=0},@var{doSimulation=false} )
## Solves for a special example the PDE's in the Nature Scientific Reports 
## article on Local Non-equilibrium information by Lee Jinwoo and Hajime Tanaka
## It does both the simulation of Brownian random walk with drift on a
## square grid and solves the PDE's, but the two are not dependent on each otherwise. 
## If the last parameter, doSimulation is true the computation time goes up by about a factor of 4 or more.
## nrounds: number of iterations. typically 5000 - 10000 makes sense
## kBT is a parameter for value for Boltzmann's const times temperature.
## zeta is a friction const..
## If deltaTime = 0 it will be computed as beta zeta /50
## The program generates a lot plots. Changed: Just one plot by default with
## 4 subplots on figure 1. 
## Newest change for acceleration: Using C-coded mexBrownianCountfile brings up 
## the speed of this computation. I make sense to run it for 20,000 iterations
## To generate use > clear mexBrownianCount; mkoctfile --mex mexBrownianCount.c
## @seealso{}
## @end deftypefn

## Author: reiner <reiner@reiner-Z97P-D3>
## Created: 2020-11-08

function [Ext, Psi, phi] = randomSquareWalk(nrounds, kBT=2, zeta=0.5, deltaTime=0.01, doSimulation=false)
  useVariableEnergyLandscape=true;
  modnumber = 10;   % plot figure 1 every modnumber;
  initialPlots=false;    % whether or not you want those pesky plots figure 1-2
  usemexFP = true;   % whether or not you want the faster mexFokkerPlanck called.
  FPrepeat=10;        % if this is 1, then deltaTime should be smaller than 0.01
  if (usemexFP)
    timescale = deltaTime;      % for proper display of iteration time
    FPdt = deltaTime/FPrepeat; % time interval subdivided. 
  else 
    timescale = deltaTime;
    FPdt = deltaTime; 
  endif
  ngrid = 80;        % careful: If doSimulation=true this may be really slow.
  beta = 1/kBT;
  baseOmega=10000;
  Nx = ngrid;
  Ny = ngrid;
  oldstylerand = false;  % never set that to true again: 10 times slower.
  [xp, yp] = meshgrid([0:1:Nx-1],[0:1:Ny-1]);
  xc = xp+0.5;
  yc = yp+0.5;    % centroids (xc,yc)
  dx = 1; dy=1;   % used in finite differences.
  noffs=12;
  scal = 1.2;
  p1 = [10,33]*scal+noffs;   s1=4*scal;    w1 = 5;
  p2 = [23,12]*scal+noffs;   s2=2.5*scal;  w2 =-4;
  p3 = [36,20]*scal+noffs;   s3=3*scal;    w3 = -2;
  p4 = [29,43]*scal+noffs;   s4=4*scal;    w4 = 4;
  p5 = [12,30]*scal+noffs;   s5=3.2*scal;  w5 = -5;
  p6 = [31,11]*scal+noffs;   s6=4.2*scal;  w6 = +3;
  cntr = [22,22]*scal+noffs;   ra=scal*20;   wa=-2.5;
  
  % compute some energy landscape:
  tauparm = 0.05;
  taustep = 1/2000;
  ExtA =  w1*exp(-0.5*((xc-p1(1)).^2+(yc-p1(2)).^2)./s1^2) ...
        +w2*exp(-0.5*(3*(xc-p2(1)).^2+(yc-p2(2)).^2)./s2^2) ...
        +w3*exp(-0.5*((xc-p3(1)).^2+2*(yc-p3(2)).^2)./s3^2) ...
        +w4*exp(-0.5*((xc-p4(1)).^2+(yc-p4(2)).^2)./s4^2) ...
        +w5*exp(-0.5*((xc-p5(1)).^2+(yc-p5(2)).^2)./s5^2) ...
        +w6*exp(-0.5*(2*(xc-p6(1)).^2+3*(yc-p6(2)).^2)./s6^2);
        
  ExtB = wa*exp(-0.05*(sqrt((xc-cntr(1)).^2 + (yc-cntr(2)).^2) - ra).^2) ...
         -w3*exp(-0.5*(0.4*(xc-p3(1)).^4+2*(yc-p3(2)).^2)./s3^2);
  %%printf("Very initial max ExtA %g  ExtB: %g \n",max(max(ExtA)),max(max(ExtB)));
  %% compute initial Energy landscape E(x,0) by interpolating   
  tauparm = gimmiSchedule(0);   % used in main loop below - for consistence. 
  Ext = (1-tauparm)*ExtA + tauparm*ExtB;
  %[vx,vy]= computeNabla(Ext,ngrid,dx,dy);
  [vx,vy] = mexNabla(Ext);
  
  if (deltaTime == 0)
   deltaTime = beta * zeta / 50 ;
  endif;
  printf("Using time steps of dt=%5.3f\n",deltaTime);
  fflush(stdout);
  % Draw energy landscape and quiver of gradient. May be overwritten
  % in the loop below.
  figure(1, 'position',[50   150 830 720]); 
  colormap(jet(256));
  
  if (initialPlots)
   colormap(jet(256));
   contour(xp,yp,Ext,40,'linewidth',2);
   hold on;
   quiver(xp,yp,-vx,-vy,2.0,'linewidth',1.5,'color','b');
   axis([0,Nx,0,Ny]);
   xlabel('x','fontsize',18); ylabel('y','fontsize',18);
   title('Energy landscape quiver {-\nabla E(x,0)}','fontsize',18);
   hold off;
   view([0, 90]);
   set(gca,'linewidth',2, 'fontsize',16);
   figure(2);
   colormap(jet(256));
   lap = mexLaplacian(Ext);
   surf(xp,yp,lap,'edgecolor','none'); 
   view([0 90]);
   xlabel('x','fontsize',18); ylabel('y','fontsize',18); 
   title('{ Energy landscape \Delta E(x,0)}','fontsize',18);
   set(gca,'linewidth',2, 'fontsize',16);
   drawnow;
   pause(2);
  endif
  noiseamplitude = sqrt(2*kBT)/zeta;                 %  so it becomes a velocity.
  xix =noiseamplitude*randn(ngrid,ngrid) - vx/zeta;   % 2nd part is drift
  xiy =noiseamplitude*randn(ngrid,ngrid) - vy/zeta;   % first part is fluctuation
  % draw an example of the driving field.
  if (initialPlots)
    figure(3);
    quiver(xp,yp,xix,xiy,1.5,'linewidth',2);
    axis([0,Nx,0,Ny]);
    xlabel('x','fontsize',18); ylabel('y','fontsize',18); 
    title('{Sample \zeta v(x,t)=-\nabla E(x,t)+\xi}','fontsize',18);
    set(gca,'linewidth',2, 'fontsize',16);
    drawnow; 
  endif;

  Pcounts = baseOmega*ones(Nx,Ny);
  
  picOne = ones(Nx,Ny);
  dt = deltaTime;
  phi = zeros(Nx,Ny);
  phizero = log(baseOmega);
  Psi = Ext; %% - phizero/beta;
  if (nrounds < 0)
    return;
  endif 
  Padd = zeros(Nx,Ny);
  lapPsi = zeros(Nx,Ny);
  dphidt = zeros(Nx,Ny);
  
  for kk = 0:nrounds;
    viewpoint = [0 90];
    if (mod(kk,modnumber)==0)
      lograt = log(Pcounts/baseOmega);
      makeSuperPlot(xp,yp,ngrid, Nx,Ny, Ext,phi,Psi,lograt,kk*timescale,tauparm,viewpoint,doSimulation);
      drawnow;
    endif  

    if (useVariableEnergyLandscape)
      tauparm = gimmiSchedule(kk);
      Ext = (1-tauparm)*ExtA + tauparm*ExtB;  %%+ ExtMouse; 
      [vx,vy]=mexNabla(Ext);
      %printf("Variable Energy land: max values %g  %g Ext %g \n", max(max(vx)), max(max(vy)), max(max(Ext)));
    endif;
    if (doSimulation)
      Pcounts = max(Pcounts - picOne, picOne); % at least one particle
      posx = xc; 
      posy = yc;
      % still have the field absdelta in here. Intention was to
      % make the strength of the fluctuation vary in space (it's const now)
      if (oldstylerand)
        xix =noiseamplitude*randn(ngrid,ngrid)-vx/zeta;  
        xiy =noiseamplitude*randn(ngrid,ngrid)-vy/zeta;
        newposx = round(xix+xc);
        newposy = round(xiy+yc);
        adx = mod(newposx,ngrid)+1;
        ady = mod(newposy,ngrid)+1;
        for k=1:ngrid, 
          for j=1:ngrid, 
            Pcounts(ady(k,j),adx(k,j)) =  Pcounts(ady(k,j),adx(k,j))+1; 
          endfor; 
        endfor;
      else 
        %[vx,vy] = mexNabla(Ext);
        %printf(" before mexBrownian:: max values %g  %g Ext %g \n", max(max(vx)), max(max(vy)), max(max(Ext)));
        Padd = mexBrownianCount(vx,vy,noiseamplitude,zeta);
        Pcounts = Pcounts+Padd;
      endif;
    endif   
   
     if (usemexFP) 
       [Psi, phi] = mexFokkerPlanck(Psi,phi,Ext,FPrepeat, FPdt, beta, zeta);
     else 
      [dvx,dvy] = mexNabla(-Psi);  
      lapPsi = mexLaplacian(Psi);   
      [dphix,dphiy] = mexNabla(phi); 
      vdotnabla = dvx.*dphix + dvy.*dphiy;
      dphidt = (lapPsi-vdotnabla)/zeta;
      phi = phi + dphidt*deltaTime;
      Psi = Ext  + phi/beta;
     endif
  endfor
endfunction

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
  
function makeSuperPlot(xp,yp,ngrid,Nx,Ny,Ext,phi,Psi,lograt,time,tauparm,viewpoint,doSimulation)
      dx = 1; 
      dy = 1;
      widx=0.48;
      widy=0.48;
      pos1 = [0.01  0.5  widx widy];
      pos2 = [0.51  0.5  widx widy];
      pos3 = [0.01  0.01 widx widy];      
      pos4 = [0.51  0.01 widx widy];

      %%set(1,'position',[40 20.000  1180 940]);
  
      titlefontsize=14;
      txtfsz=14;
      % 
      
      subplot(2,2,1,'align');
 
      titletext = maketitleString(Ext, "E(x,t)",time);
      Ext(1,1) = -3;
      Ext(Nx,Ny)= 3;
      surf(xp,yp,Ext,'edgecolor','none');
      xlabel('x','fontsize',txtfsz); ylabel('y','fontsize',txtfsz); zlabel('E(x,t)','fontsize',txtfsz);
      title(sprintf("%s {\\tau=%5.3f}",titletext, tauparm),'fontsize',titlefontsize);
      view(viewpoint);
      set(gca,'linewidth',2, 'fontsize',16);
      axis('off');
      
      subplot(2,2,2,'align'); 
      %subplot(2,2,2,'position',pos2);
      laplacePsi = mexLaplacian(Psi); 
      titletext = maketitleString(laplacePsi,"{\\Delta \\Psi}(x,t) ",time);
      laplacePsi(1,1)=-0.025;
      laplacePsi(Nx,Ny)=0.025;
      surf(xp,yp,laplacePsi,'edgecolor','none');

      axis([0,Nx,0,Ny]);
      xlabel('x','fontsize',txtfsz); ylabel('y','fontsize',txtfsz);
      title(titletext,'fontsize',titlefontsize);
      view(viewpoint);
      %hold off;
      set(gca,'linewidth',2, 'fontsize',16);
      axis('off');

      subplot(2,2,3,'align'); 
      if (doSimulation)
        titletext = maketitleString(lograt,"{Log(\\Omega(\\Lambda^x)/\\Omega(\\Lambda))}", time);
        lograt(1,1)=0.025;
        lograt(Nx,Ny)=-0.025;
        surf(xp,yp,lograt,'edgecolor','none');
        xlabel('x','fontsize',txtfsz); ylabel('y','fontsize',txtfsz); zlabel('{log(\Omega(\Lambda^x))}','fontsize',txtfsz);
        title(titletext,'fontsize',titlefontsize);
        view(viewpoint);
        set(gca,'linewidth',2, 'fontsize',16);
        axis('off');
      else 
        titletext = maketitleString(Psi,"Free energy {\\Psi}", time);
        Psi(1,1)=1;
        Psi(Nx,Ny)=-1;
        surf(xp,yp,Psi,'edgecolor','none');
        xlabel('x','fontsize',txtfsz); 
        ylabel('y','fontsize',txtfsz); 
        zlabel('{\Psi(x,t)}','fontsize',txtfsz);
        title(titletext,'fontsize',titlefontsize);
         view(viewpoint);
        set(gca,'linewidth',2, 'fontsize',16);
        axis('off');
      endif;  
      
      subplot(2,2,4,'align'); 
      %subplot(2,2,4,'position',pos4);
      titletext = maketitleString(phi,"Local Information {\\phi}", time);  
      phi(1,1)=1/2;
      phi(Nx,Ny)=-1/2;
      surf(xp,yp, phi,'edgecolor','none');
      xlabel('x','fontsize',txtfsz); ylabel('y','fontsize',txtfsz); zlabel('{\phi(x,t)}','fontsize',txtfsz);
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
%  
%function [vx, vy] = computeNabla(F,ngrid,dx,dy);
%opx = [-1 8 0 -8 1]/(12*dx);
%opy = [-1 8 0 -8 1]/(12*dy);
%vx = conv2(F,  opx,  'same'); 
%vy = conv2(F,  opy', 'same');  % note the transpose.
%endfunction;
%%
%function [divV] = computeDiv(F,ngrid,dx,dy);
%opx = [-1 8 0 -8 1]/(12*dx);
%opy = [-1 8 0 -8 1]/(12*dy);
%[vx, vy] = computeNabla(F,ngrid,dx,dy);
%divV = conv2(vx, opx, 'same') + conv2(vy,opy','same');
%endfunction;

%function [lap] = computeLaplacian(F,ngrid);
%  K = [0    0  -1/12    0    0; 
%       0    0    4/3    0    0 ; 
%    -1/12  4/3  -5    4/3  -1/12; 
%       0    0    4/3    0    0; 
%       0    0   -1/12   0    0];
%  lap = conv2(F,K,'same');
%endfunction;
%
%function [lap] = computeLaplacian(F,ngrid);
%  c1=0; c2=-1/30; c3=-1/60; c4=4/15; c5=13/15; c6=-21/5;
%  K = [c1,  c2, c3, c2, c1;
%       c2,  c4, c5, c4, c2;
%       c3,  c5, c6, c5, c3;
%       c2,  c4, c5, c4, c2;
%       c1,  c2, c3, c2, c1];
%  lap = conv2(F,K,'same');
%endfunction;