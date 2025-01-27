import Link from "next/link";
export default function Home() {
  return (
    <section className="h-screen min-w-screen flex flex-col items-center justify-center gap-5">
      <div className="text-4xl">
          MLOpt
      </div>
      <div className="flex items-center justify-center flex-col">
        <Link href={'/dashboard'}>Start Preprocessing</Link>
      </div>
    </section>
  );
}
