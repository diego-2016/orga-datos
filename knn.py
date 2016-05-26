from sklearn import neighbors
from numpy import genfromtxt, savetxt

def knn(k,p):
	print "obteniendo datos de dataset train..."
	dataset = genfromtxt(open("/home/diego/Descargas/train.csv","rb"), delimiter=",", dtype="f8")[1:]    
	target = [x[0] for x in dataset]
	train = [x[1:] for x in dataset]
	print "obteniendo datos de dataset test..."
	test = genfromtxt(open("/home/diego/Descargas/test.csv","rb"), delimiter=",", dtype="f8")[1:]
	print "aplicando knn ..."
	#k=3 pesos=uniformes espacio de particiones de division de datos=kd_tree 
    #tamanio de los nodos hojas del kd_tree=30 parametro p(de distancia minkowsky=2)
    #probar con distancia manhattan p=1 y distancias p<0 ej: p=0.5 p=.0.025 p=0.7
	knn = neighbors.KNeighborsClassifier(k,"uniform","kd_tree",30,p)
	knn.fit(train, target)
	predictions = knn.predict(test)
	print "guardando prediccion en archivo..."
	#creo archivo vacio
	archivo_prediccion=open("/home/diego/Documentos/Facultad/Orga-datos/prediction knn-P es "+str(p)+"-K es"+str(k)+".csv","w")
	archivo_prediccion.close()
	#lo abro para escribir
	archivo_prediccion=open("/home/diego/Documentos/Facultad/Orga-datos/prediction knn-P es "+str(p)+"-K es"+str(k)+".csv","a")
	#savetxt("/home/diego/Documentos/Facultad/Orga-datos/submissionKnn"+"-k es "+str(k)+"- p es "+str(p)+".csv", predictions, delimiter=",", fmt="%d")
	archivo_prediccion.write('ImageId,Label'+'\n')
	indice=0
	for dato_test,prediccion_clase in zip(test,predictions):
		indice=indice+1
		#print "va a escribir el dato archivo en iteracion:"+str(cont)
		archivo_prediccion.write(str(indice)+','+str(prediccion_clase)+'\n')
	archivo_prediccion.close()
	print "fin"
		
knn(k=7,p=1)
