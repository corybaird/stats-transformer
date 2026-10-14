% DATA.M

% Quarterly data from FRED
% Gross National Product: Chain-type Price Index: Index 2005=100: SA, Q
% Real Gross Domestic Product: Billions of Chained 2005 Dollars: SAAR, Q
% Effective federal funds rate, M
% 1954.IV-2007.IV (exclude unconventional monetary policy after 2007.Q4)

% Sampple reduced to 1959.II-2007.IV
load gnpdeflator.txt; infl=dif(log(gnpdeflator(:,3)))*100;
defl=log(gnpdeflator(:,3));

load realgnp.txt; drgdp=dif(log(realgnp(:,3)))*100;
rgdp=log(realgnp(:,3));

load fedfunds.txt;
irate=[];
for i=1:3:length(fedfunds(:,3))
  irate=[irate; mean(fedfunds(i:i+2,3))];
end;    

% 1959.I-2007.IV reduced to 1959.II-2007.IV
load m1.txt;
m1=log(m1(3:end,3));
dmoney=[];
for i=4:3:length(m1)
    dmoney=[dmoney; (m1(i,1)-m1(i-3,1))*100];
end;

money=[];
for i=1:3:length(m1)
    money=[money; m1(i,1)];
end;

% Data for Keating (1992) VAR model
y=[infl(19:end,1) drgdp(19:end,1) irate(19:end,1) dmoney];


