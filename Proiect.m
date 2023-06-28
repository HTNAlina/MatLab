%adaugă calea către directorul care conține pachetul "statistics_and_machine_learning_toolbox" 
addpath('C:\Program Files\MATLAB\R2023a\toolbox\stats');

%import clasele din Matlab
import matlab.io.datastore.*
import statistics.*

%incarcare dataset
data=readtable('D:\LucrareLicenta\Predictia_diabetului.csv');

%import clasa StandardScaler pentru scalarea datelor
import sklearn.preprocessing.StandardScaler.*

%import pachetulSVM pentru clasificarea datelor
import compactSVM.*
import sklearn.metrics.accuracy_score.*

%info despre date : 
summary(data);
%afisare cap de tabela:  head(data);
%numarul de aparitii ale unei valori in Rezultat: tabulate(data.Rezultat);

%Calculam media variabilelor din setul de date în funcție de valorile din coloana "Rezultat".
stats=grpstats(data,{'Rezultat'},"mean");
%disp(stats);

%Separam coloana Rezultat de restul datelor Y-coloana rezultat X-celelate
%date
X = data(:, 1:end-1);
Y = data(:, end);
%disp(X);
%disp(Y);

% Convertim X în matrice numerică
X = table2array(X);

% Definim dimensiunile subsetului de testare
% 80% antrenare si 20% testare
test_size = 0.2;

% Amestecarea și împărțirea setului de date în subseturi de antrenare și testare
% Se generează un vector aleator de permutări pentru a reordona indicii rândurilor setului de date
idx = randperm(size(X, 1));
num_test = round(test_size * size(X, 1));
idx_test = idx(1:num_test);
idx_train = idx(num_test+1:end);

X_train = X(idx_train, :);
X_test = X(idx_test, :);
Y_train = Y(idx_train, :);
Y_test = Y(idx_test, :);

% Afișarea dimensiunilor matricelor
%disp(size(X));
%disp(size(X_train));
%disp(size(X_test));

% Convertim etichetele într-o matrice numerică
Y_train = table2array(Y_train);
Y_test = table2array(Y_test);

% Definim și antrenăm clasificatorul SVM
classifier = fitcsvm(X_train, Y_train, 'KernelFunction', 'linear');

% Realizăm predicția pe setul de antrenare
X_train_prediction = predict(classifier, X_train);

% Calculăm acuratețea setului de antrenare pentru a evalua performanța
% pentru datele oferite
training_data_accuracy = sum(X_train_prediction == Y_train) / numel(Y_train);

% Afișăm acuratețea setului de antrenare
%disp(['Acuratetea setului de antrenare: ', num2str(training_data_accuracy)]);

% Realizăm predicția pe setul de testare
X_test_prediction = predict(classifier, X_test);

% Calculăm acuratețea setului de testare
test_data_accuracy = sum(X_test_prediction == Y_test) / numel(Y_test);

% Afișăm acuratețea setului de testare
%disp(['Acuratetea setului de testare: ', num2str(test_data_accuracy)]);

%input_data reprezintă un set de date de intrare 
%input_data = [59, 1, 189, 60, 23, 30];

% Citirea valorilor de la tastatură
% v1 = input('Varsta: ');
%v2 = input('Sarcina: ');
%v3 = input('Glucoza: ');
%v4 = input('Tensiune: ');
%v5 = input('Insulina: ');
%v6 = input('IMC(kg/m^2): ');

run('C:\Users\blasc\Downloads\D_LucrareLicentapredictii.m');

% Acum puteți utiliza variabilele varsta, sarcina, glucoza, tensiune, insulina și imc în codul dvs. MATLAB
v1 = varsta;
v2 = sarcina;
v3 = glucoza;
v4 = tensiune;
v5 = insulina;
v6 = imc;

% Crearea vectorului input_data
input_data = [v1, v2, v3, v4, v5, v6];

%predictie
prediction = predict(classifier, input_data);
disp(['Rezultat: ', num2str(prediction)]);

if prediction == 0
    disp('Persoana nu are diabet');
else
    disp('Persoana are diabet');
end


% Specify the path to your text file
filePath = 'C:\Users\blasc\Desktop\DiabetPredictie_Lic\src\prediction.json';

dlmwrite(filePath, prediction); % Salvarea valorii predicției într-un fișier text

delete('C:\Users\blasc\Downloads\D_LucrareLicentapredictii.m')

