function testSigmoidGradient()
% testSigmoidGradient unit test for sigmoidGradient function

    rowVector = [-1.000000000000000,-0.500000000000000,0.000000000000000,0.500000000000000,...
                    1.000000000000000];
    colvector = rowVector';
    aMatrix = [colvector colvector];

    scalarResult = sigmoidGradient(0.25);
    rowVectorResult = sigmoidGradient(rowVector);
    colvectorResult = sigmoidGradient(colvector);
    matrixResult = sigmoidGradient(aMatrix);

    
    expectedScalarResult = 0.250000000000000;
    expectedRowResult = [0.196611933241482, 0.235003712201594, 0.250000000000000, 0.235003712201594, ...
                             0.196611933241482];
    expectedColResult = expectedRowResult';
    expectedMatrixResult = [expectedColResult expectedColResult];

    assert(scalarResult, expectedScalarResult, "Scalar results do not match");
    assert(rowVectorResult, expectedRowResult);
    assert(colvectorResult, expectedColResult, "Column vectors results do not match");
    assert(matrixResult, expectedMatrixResult);
end

