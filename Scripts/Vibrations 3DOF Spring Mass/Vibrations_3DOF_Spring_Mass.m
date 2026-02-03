% Camden Hill
% Vibrations 3DOF Mass Damper
format longg

syms w t

q0 = [1; -0.5; -1.2];
q0_ = [0; 0; 0];
M = [4 0 0; 0 2 0; 0 0 6];
K = [4, -1, 0; -1, 3, -2; 0, -2, 2];
disp("M =")
disp(M)
disp("K =")
disp(K)
A = K - ((w)*M);
Adet = det(A);
disp("Characteristic Equation:")
disp(Adet)
% Calculate the eigenvalues and eigenvectors of matrix A
eigenValues = vpasolve(Adet == 0,w);
eigenValues_real = zeros(1,height(M));
for i = 1:height(M)
    eigenValues_real(i) = eigenValues(i);
end
eigenValues = eigenValues_real;
eigenVectors = zeros(height(M),height(M));
for i = 1:height(M)
    eigenVectors(:,i) = null(double(subs(A, w, eigenValues(i)))); % Calculate eigenvectors
end
normalized_check = [norm(eigenVectors(:,1)), norm(eigenVectors(:,2)), norm(eigenVectors(:,3))];
% Display the eigenvalues and eigenvectors
disp('Eigenvalues:');
disp(eigenValues);
disp("Check of Eigenvalues:")
disp([double(subs(Adet, w, eigenValues(1))), double(subs(Adet, w, eigenValues(2))), double(subs(Adet, w, eigenValues(3)))])
disp('Eigenvectors:');
disp(eigenVectors);
disp("Check if Eigenvectors are normalized:")
disp(normalized_check)

disp("Check if Eigenvectors diagonalize Inertial Matrix:")
disp(eigenVectors.'*M*eigenVectors)
disp("Check if Eigenvectors diagonalize Stiffness Matrix:")
disp(eigenVectors.'*K*eigenVectors)
disp("Check if Individual Eigenvectors diagonalize:")
disp([eigenVectors(:,1).'*M*eigenVectors(:,1),eigenVectors(:,2).'*M*eigenVectors(:,2)])
disp([eigenVectors(:,1).'*K*eigenVectors(:,1),eigenVectors(:,2).'*K*eigenVectors(:,2)])
Omegasq = (eigenVectors.'*K*eigenVectors) * (inv(eigenVectors.'*M*eigenVectors));
disp("Modal Diagonal Matrix =")
disp(Omegasq)
eta0 = eigenVectors\q0; % inv(eigenVectors) * q0; eigenVectors\q0;
eta_0 = eigenVectors\q0_; % inv(eigenVectors) * q0_; eigenVectors\q0_;
disp("Eta0 or Initial Modal Displacements =")
disp(eta0)
disp("Eta_0 or Initial Modal Velocity =")
disp(eta_0)
q_eqs_consts = sym('q_eqs_consts',[height(M),2]);
for i = 1:height(M)
    q_eqs_consts(i,:) = [eta0(i)*cos(eigenValues(i).*t), (eta_0(i)./eigenValues(i))*sin(eigenValues(i).*t)]; % (eta0(i).*cos(eigenValues(i).*t)) + ((eta_0(i)./eigenValues(i)).*sin(eigenValues(i).*t));
end
disp("q Equations Constants =")
disp(q_eqs_consts)
q_eqs_matmul = eigenVectors*q_eqs_consts;
disp("q Equations Matrix Multiplied =")
disp(q_eqs_matmul)
q_eqs = sym('q_eqs',[height(M),1]);
for i = 1:height(M)
    q_eqs(i,1) = q_eqs_matmul(i,1) + q_eqs_matmul(i,2);
end
disp("q Equations =")
disp(q_eqs)
disp("q Equations w/ 4 digits =")
disp(vpa(q_eqs,4))

% Missing last mode only
Mode_num_analyzed = height(M) - 2;
tvals = linspace(0,10,201).';
q_eqs_mod = sym('q_eqs_mod',[height(M),1]);
for i = 1:height(M)
    q_eqs_mod(i) = 0;
end
for i = 1:height(M)
    Amat = zeros(201,2*Mode_num_analyzed);
    bvect = double(subs(q_eqs(i),t,tvals));
    for j = 1:Mode_num_analyzed
        Amat(:,(2*(j-1))+1:(2*(j-1))+2) = [cos(eigenValues(j)*tvals),sin(eigenValues(j)*tvals)];
    end
    Amat = double(Amat);
    lstqrs_consts = lsqr(Amat,bvect);
    disp("Lstqrs Fit Constants =")
    disp(lstqrs_consts)
    for j = 1:Mode_num_analyzed
        q_eqs_mod(i) = q_eqs_mod(i) + (lstqrs_consts((2*(j-1))+1)*cos(eigenValues(j)*t)) + (lstqrs_consts((2*(j-1))+2)*sin(eigenValues(j)*t));
    end
end
disp("q Equations Modified by Least-Squares =")
disp(q_eqs_mod)
disp(vpa(q_eqs_mod,2))
% Mode_num_analyzed = height(M) - 1;
% IC_mat = horzcat(q0,q0_);
% q0_mod = eigenVectors(:, 1:Mode_num_analyzed)*q_eqs_consts(1:Mode_num_analyzed,:);
% disp("q Components Modified =")
% disp(q0_mod)
% q0_mat = double(subs(q0_mod,t,0));
% disp("q0 Components Modified =")
% disp(q0_mat)
% q0bar = sym('q0bar',[height(M),1]);
% for j = 1:height(M)
%     q0bar(j,1) = q0_mat(j,1) + q0_mat(j,2);
% end
% disp("q0_bar Equation =")
% disp(double(q0bar))
% 
% q0_mod = eigenVectors(:, 1:Mode_num_analyzed)*diff(q_eqs_consts(1:Mode_num_analyzed,:),t);
% disp("q' Components Modified =")
% disp(q0_mod)
% q0_mat = double(subs(q0_mod,t,0));
% disp("q0' Components Modified =")
% disp(q0_mat)
% q0_bar = sym('q0bar',[height(M),1]);
% for j = 1:height(M)
%     q0_bar(j,1) = q0_mat(j,1) + q0_mat(j,2);
% end
% disp("q0_bar' Equation =")
% disp(double(q0_bar))
% 
% q0_bar_combined_mat = horzcat(q0bar,q0_bar);
% disp("q_bar Matrix =")
% disp(q0_bar_combined_mat)
% weight_matrix = ((q0_bar_combined_mat.'*q0_bar_combined_mat)^-1)*q0_bar_combined_mat.'*IC_mat;
% disp("Weight Matrix =")
% disp(weight_matrix)



clear()