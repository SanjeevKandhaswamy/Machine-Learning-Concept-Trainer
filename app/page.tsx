import Link from "next/link"
import { ArrowRight, BookOpen, Code, Database, LineChart } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"

export default function Home() {
  const concepts = [
    {
      title: "Naive Bayes",
      description: "Probabilistic classifiers based on applying Bayes' theorem with strong independence assumptions.",
      href: "/naive-bayes",
      icon: <Database className="h-5 w-5" />,
    },
    {
      title: "K-Nearest Neighbors",
      description: "A non-parametric method used for classification and regression.",
      href: "/knn",
      icon: <Database className="h-5 w-5" />,
    },
    {
      title: "Decision Trees",
      description: "A decision support tool that uses a tree-like model of decisions and their possible consequences.",
      href: "/decision-trees",
      icon: <Database className="h-5 w-5" />,
    },
    {
      title: "Linear Regression",
      description:
        "A linear approach to modeling the relationship between a dependent variable and one or more independent variables.",
      href: "/linear-regression",
      icon: <LineChart className="h-5 w-5" />,
    },
    {
      title: "Logistic Regression",
      description: "A statistical model that uses a logistic function to model a binary dependent variable.",
      href: "/logistic-regression",
      icon: <LineChart className="h-5 w-5" />,
    },
    {
      title: "Random Forests",
      description:
        "An ensemble learning method for classification, regression and other tasks that operates by constructing multiple decision trees.",
      href: "/random-forests",
      icon: <Database className="h-5 w-5" />,
    },
  ]

  return (
    <div className="flex flex-col min-h-screen">
      <header className="sticky top-0 z-10 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-14 items-center">
          <div className="mr-4 flex">
            <Link href="/" className="mr-6 flex items-center space-x-2">
              <BookOpen className="h-6 w-6" />
              <span className="font-bold">ML Concepts Trainer</span>
            </Link>
          </div>
          <nav className="flex items-center space-x-6 text-sm font-medium">
            <Link href="https://ml-resources.vercel.app/" className="transition-colors hover:text-foreground/80 text-muted-foreground">
              Resources
            </Link>
          </nav>
        </div>
      </header>
      <main className="flex-1">
        <section className="w-full py-12 md:py-24 lg:py-32 bg-muted/40">
          <div className="container px-4 md:px-6">
            <div className="flex flex-col items-center justify-center space-y-4 text-center">
              <div className="space-y-2">
                <h1 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl lg:text-6xl">
                  Learn Machine Learning Interactively
                </h1>
                <p className="mx-auto max-w-[700px] text-muted-foreground md:text-xl">
                  Master key machine learning algorithms through theory, real-world applications, and hands-on coding
                  exercises.
                </p>
              </div>
              <div className="space-x-4">
                <Button asChild>
                  <Link href="#concepts">
                    Get Started
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Link>
                </Button>
              </div>
            </div>
          </div>
        </section>
        <section id="concepts" className="container py-12 md:py-24 lg:py-32">
          <div className="mx-auto grid justify-center gap-4 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-3">
            {concepts.map((concept) => (
              <Card key={concept.title} className="flex flex-col">
                <CardHeader>
                  <div className="flex items-center gap-2">
                    {concept.icon}
                    <CardTitle>{concept.title}</CardTitle>
                  </div>
                  <CardDescription>{concept.description}</CardDescription>
                </CardHeader>
                <CardContent className="flex-1">
                  <div className="flex items-center gap-2">
                    <Code className="h-4 w-4 text-muted-foreground" />
                    <span className="text-sm text-muted-foreground">Interactive coding exercises</span>
                  </div>
                </CardContent>
                <CardFooter>
                  <Button asChild className="w-full">
                    <Link href={concept.href}>
                      Learn {concept.title}
                      <ArrowRight className="ml-2 h-4 w-4" />
                    </Link>
                  </Button>
                </CardFooter>
              </Card>
            ))}
          </div>
        </section>
        <section className="w-full py-12 md:py-24 lg:py-32 bg-muted/40">
          <div className="container px-4 md:px-6">
            <div className="flex flex-col items-center justify-center space-y-4 text-center">
              <div className="space-y-2">
                <h2 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl">Why Learn ML here?</h2>
                <p className="mx-auto max-w-[700px] text-muted-foreground md:text-xl">
                  Our interactive approach combines theory with practice to help you truly understand machine learning
                  concepts.
                </p>
              </div>
              <div className="mx-auto grid max-w-5xl items-center gap-6 py-12 lg:grid-cols-3">
                <div className="flex flex-col items-center space-y-2 border rounded-lg p-6 bg-background">
                  <BookOpen className="h-12 w-12 text-primary" />
                  <h3 className="text-xl font-bold">Comprehensive Theory</h3>
                  <p className="text-center text-muted-foreground">
                    Detailed explanations of each algorithm with mathematical foundations.
                  </p>
                </div>
                <div className="flex flex-col items-center space-y-2 border rounded-lg p-6 bg-background">
                  <Database className="h-12 w-12 text-primary" />
                  <h3 className="text-xl font-bold">Real-world Applications</h3>
                  <p className="text-center text-muted-foreground">
                    Learn how these algorithms are applied to solve actual problems.
                  </p>
                </div>
                <div className="flex flex-col items-center space-y-2 border rounded-lg p-6 bg-background">
                  <Code className="h-12 w-12 text-primary" />
                  <h3 className="text-xl font-bold">Interactive Coding</h3>
                  <p className="text-center text-muted-foreground">
                    Write and execute code directly in your browser with immediate feedback.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>
      <footer className="border-t py-6 md:py-0">
        <div className="container flex flex-col items-center justify-between gap-4 md:h-24 md:flex-row">
          <p className="text-center text-sm leading-loose text-muted-foreground md:text-left">
            © {new Date().getFullYear()} ML Academy. All rights reserved.
          </p>
          <div className="flex items-center gap-4">
            <Link href="#" className="text-sm text-muted-foreground underline-offset-4 hover:underline">
              Terms
            </Link>
            <Link href="#" className="text-sm text-muted-foreground underline-offset-4 hover:underline">
              Privacy
            </Link>
          </div>
        </div>
      </footer>
    </div>
  )
}
