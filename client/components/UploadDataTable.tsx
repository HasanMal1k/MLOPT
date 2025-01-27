import {
    Table,
    TableBody,
    TableCaption,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
  } from "@/components/ui/table"
  
import { X } from "lucide-react"  

type data = {
  files: File[];
  setFiles: React.Dispatch<React.SetStateAction<File[]>>;
}


function UploadedDataTable({files, setFiles} : data) {

  const removeItem = (files: File[], fileName: string) => {
    const updatedFiles = files.filter((item)=>{ item.name !== fileName })
    setFiles(updatedFiles)
  }

  return (
    <Table className="mt-10">
      <TableCaption>Uploaded Data</TableCaption>
      <TableHeader>
        <TableRow>
          <TableHead>Name</TableHead>
          <TableHead>Size</TableHead>
          <TableHead>Format</TableHead>
          <TableHead className="text-right">Remove</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
          {files.map((item, index)=>(
          <TableRow key={index}>
            <TableCell className="font-semibold">{item.name}</TableCell>
            <TableCell className="font-semibold">{(item.size / 1048576).toFixed(1)}</TableCell> 
            <TableCell className="font-semibold">{item.type}</TableCell>
            <TableCell className="text-right"><button onClick={()=>{removeItem(files, item.name)}}><X color="gray"/></button></TableCell> 
            </TableRow>
          ))}
          {/* <TableCell className="font-medium">Data Name</TableCell>
          <TableCell>Size</TableCell>
          <TableCell>CSV / XLSX</TableCell>
          <TableCell className="text-right"><button><X color="gray"/></button></TableCell> */}
      </TableBody>
    </Table>
  )
}

export default UploadedDataTable  