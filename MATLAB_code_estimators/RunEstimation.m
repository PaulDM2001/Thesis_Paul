%%RUN ESTIMATION ROUTINE
% Kamil Kladivko for Technical Computing Prague 2007

clc
clear all
close all

url = 'https://fred.stlouisfed.org/';
c = fred(url);
series_base_rate = 'DGS1';
series_spread_BBB = 'BAMLC0A4CBBB';
series_spread_A = 'BAMLC0A3CA';
series_spread_AA = 'BAMLC0A2CAA';
series_spread_AAA = 'BAMLC0A1CAAA';

startdate = '01/01/2023'; 
enddate = '01/10/2025';
base_rate = fetch(c, series_base_rate, startdate, enddate);
spread_BBB = fetch(c, series_spread_BBB, startdate, enddate);
spread_A = fetch(c, series_spread_A, startdate, enddate);
spread_AA = fetch(c, series_spread_AA, startdate, enddate);
spread_AAA = fetch(c, series_spread_AAA, startdate, enddate);
close(c)

figure(1)
hold on 
plot(base_rate.Data(:, 2)./100)
plot(spread_BBB.Data(:, 2)./100)
plot(spread_A.Data(:, 2)./100)
plot(spread_AA.Data(:, 2)./100)
plot(spread_AAA.Data(:, 2)./100)
hold off

Model.Data = rmmissing(base_rate.Data(:, 2)) ./100;
Model.TimeStep = 1/250;     % recommended: 1/250 for daily data, 1/12 for monthly data, etc
Model.Disp = 'y';           % 'y' | 'n' (default: y)
Model.MatlabDisp = 'off';  % 'off'|'iter'|'notify'|'final'  (default: off)
Model.Method = 'besseli';   % 'besseli' | 'ncx2pdf' (default: besseli)

fprintf('Results Base')
Results_base = CIRestimation(Model);

fprintf('Results BBB')
Model.Data = rmmissing(spread_BBB.Data(:, 2)) ./100;
Results_spread_BBB = CIRestimation(Model);

fprintf('Results A')
Model.Data = rmmissing(spread_A.Data(:, 2)) ./100;
Results_spread_A = CIRestimation(Model);

fprintf('Results AA')
Model.Data = rmmissing(spread_AA.Data(:, 2)) ./100;
Results_spread_AA = CIRestimation(Model);

fprintf('Results AAA')
Model.Data = rmmissing(spread_AAA.Data(:, 2)) ./100;
Results_spread_AAA = CIRestimation(Model);