import Link from "next/link";
import { Button } from "@/components/ui/button";
export default function Home() {
  return (
    <section className="h-screen min-w-screen flex items-center justify-center gap-5">
      {/* Title Veghera */}
      <div className="h-full w-1/2 flex flex-col gap-5 items-start justify-center px-20 mb-40">
         <h1 className="text-6xl font-bold">Welcome To MLOpt</h1>
         <p className="text-lg font-normal tracking-wide">Reducing your machine learning workflow to clicks</p>
          <Button><Link href={'/dashboard'}> Continue To Dashboard</Link></Button>
      </div>

      <div className="h-full w-1/2 bg-white">

      </div>
    </section>
  );
}



// Pehla sirf ye tha

{/* <div className="text-4xl">
          MLOpt
      </div>
      <div className="flex items-center justify-center flex-col">
        <Link href={'/dashboard'}>Start Preprocessing</Link>
      </div> */}