library(shiny)

shinyServer
(
  function(input,output){

    result<-eventReactive(input$Submit,
                          {
                            
                            ha<-matrix(ncol = 11,c(as.numeric(input$age),as.numeric(input$sex),as.numeric(input$cp),as.numeric(input$trestbps),as.numeric(input$chol),as.numeric(input$fbs),as.numeric(input$restecg),as.numeric(input$thalach),as.numeric(input$exang),as.numeric(input$oldpeak),as.numeric(input$slope)))
                            colnames(ha)<-c("age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope")
                            k<-as.data.frame(ha)
                            pred_system<-predict(logRegModel,k,method="class",type="prob")
                            pred_result<-as.data.frame(pred_system)
                            pred_result$positive
                          })
    output$text<-renderPrint({paste("The probrability of heart disease :",result())})
  }
)